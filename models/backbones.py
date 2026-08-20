import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection, Mask2FormerModel, SamModel


class CrossAttnBlock(nn.Module):
    """Post-LN cross-attention with residual. Used to inject token-level
    text + box conditioning into the Mask2Former queries.

    Follows the DETR convention: query_pos (if given) is added to the
    attention Q only, not to the residual path."""

    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, query_pos=None, key_padding_mask=None):
        q = query if query_pos is None else query + query_pos
        attn_out, _ = self.attn(
            q, key_value, key_value,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(query + self.dropout(attn_out))


class CustomSegmentationModel(nn.Module):
    """
    Variant of models_v3_debug_segment_debug that fixes the conditioning
    pooling bug.

    Changes vs base:
      - Text uses the full token sequence (B, T, C) rather than the pooled
        text_embeds. Preserves per-token semantics needed for zero-shot
        class names.
      - Box keeps the two SAM corner tokens as a (B, 2, C) sequence rather
        than mean-pooling them. Averaging the Fourier-style positional
        embeddings of the two corners destroys corner identity and the
        box size/aspect signal — only an artifact of the center remained.
      - Queries cross-attend to the concatenated [text; box] token
        sequence before the Mask2Former transformer decoder, replacing
        the broadcast-add of a single global cond vector.
    """

    def __init__(self,
                 base_model_name,
                 d_model=None,
                 nhead=8,
                 num_layers=None,
                 bbx_random=0.0,
                 tune_mode='freeze',
                 mask2former_name='facebook/mask2former-swin-small-ade-semantic',
                 num_queries=20,
                 image_size=512,
                 sam_pretrained='facebook/sam-vit-base'):
        super().__init__()
        self.bbx_random = bbx_random

        # PLIP (CLIP) text encoder — used as a token-level conditioning
        # signal. Image encoder is intentionally not loaded.
        self.base_model = CLIPTextModelWithProjection.from_pretrained(base_model_name)
        if tune_mode == 'freeze':
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Mask2Former: Swin encoder + pixel decoder + transformer decoder.
        m2f = Mask2FormerModel.from_pretrained(mask2former_name)
        self.m2f_encoder        = m2f.pixel_level_module.encoder
        self.pixel_decoder      = m2f.pixel_level_module.decoder
        self.transformer_module = m2f.transformer_module
        m2f_hidden_dim          = self.transformer_module.queries_features.embedding_dim
        del m2f

        # Project the CLIP text token sequence into the M2F hidden dim.
        # CLIPTextModelWithProjection's last_hidden_state is in
        # config.hidden_size (pre-projection text-transformer width).
        text_hidden = self.base_model.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, m2f_hidden_dim)

        # SAM bbox prompt encoder, frozen. We keep `prompt_encoder` only;
        # the SAM vision encoder and mask decoder are discarded.
        sam = SamModel.from_pretrained(sam_pretrained)
        self.prompt_encoder = sam.prompt_encoder
        self.prompt_encoder.input_image_size = image_size
        del sam

        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

        # Per-corner-token projection: SAM hidden -> M2F hidden.
        sam_hidden = self.prompt_encoder.hidden_size
        self.box_proj = (
            nn.Identity() if sam_hidden == m2f_hidden_dim
            else nn.Linear(sam_hidden, m2f_hidden_dim)
        )

        # Two learnable no-box tokens. Mirrors the K=2 corner-token
        # structure produced by SAM's prompt encoder so the cross-attn
        # sees the same sequence length in both "has box" and "no box"
        # branches. Small random init so absence is a distinguishable
        # signal from the start (zero init collapses through MHA's
        # W_k/W_v and leaves attention to the bias only).
        self.no_box_embed = nn.Embedding(2, m2f_hidden_dim)
        nn.init.normal_(self.no_box_embed.weight, std=0.02)

        # Query <- [text_tokens ; box_tokens] cross-attention. Replaces
        # the pooled-vector broadcast-add used by the base variant.
        self.cond_attn = CrossAttnBlock(d_model=m2f_hidden_dim, nhead=nhead)

        # Resize 100-query embeddings to num_queries by KEEPING the first
        # `num_queries` rows of the pretrained embeddings.
        self.num_queries = num_queries
        old_emb  = self.transformer_module.queries_embedder
        old_feat = self.transformer_module.queries_features
        assert num_queries <= old_emb.num_embeddings, (
            f"num_queries={num_queries} exceeds pretrained "
            f"{old_emb.num_embeddings}; cannot slice."
        )
        self.transformer_module.queries_embedder = nn.Embedding.from_pretrained(
            old_emb.weight[:num_queries].clone(), freeze=False,
        )
        self.transformer_module.queries_features = nn.Embedding.from_pretrained(
            old_feat.weight[:num_queries].clone(), freeze=False,
        )

        # Per-query binary classification head: (bg, fg).
        self.class_head = nn.Linear(m2f_hidden_dim, 2)

    def _build_cond_tokens(self, text_tokens, text_pad_mask, box_tokens):
        """Concat text + box token sequences and build a joint
        key_padding_mask (True = ignore)."""
        cond_tokens = torch.cat([text_tokens, box_tokens], dim=1)  # (B, T+K, C)
        B, K, _ = box_tokens.shape
        box_pad_mask = torch.zeros(
            B, K, dtype=torch.bool, device=box_tokens.device,
        )
        pad_mask = torch.cat([text_pad_mask, box_pad_mask], dim=1)
        return cond_tokens, pad_mask

    def _run_transformer_with_cond(self, multi_scale_features, mask_features,
                                   cond_tokens, cond_pad_mask):
        """Replica of HF Mask2FormerTransformerModule.forward, but with a
        cross-attention from queries to (text + box) token sequence
        applied before the masked-attention decoder."""
        tm = self.transformer_module

        multi_stage_features, multi_stage_pos_embeds, size_list = [], [], []
        for i in range(tm.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_pos_embeds.append(
                tm.position_embedder(multi_scale_features[i], None).flatten(2)
            )
            multi_stage_features.append(
                tm.input_projections[i](multi_scale_features[i]).flatten(2)
                + tm.level_embed.weight[i][None, :, None]
            )
            multi_stage_pos_embeds[-1] = multi_stage_pos_embeds[-1].permute(2, 0, 1)
            multi_stage_features[-1]  = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # (Q, B, C)
        query_pos  = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_feat = tm.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Cross-attn over the cond token sequence. Switch to batch-first
        # for the attention call, then back to (Q, B, C). query_pos is
        # added to the attention Q only (DETR-style), not to the residual.
        q_bqc   = query_feat.permute(1, 0, 2)
        qpos_bqc = query_pos.permute(1, 0, 2)
        q_bqc = self.cond_attn(
            q_bqc, cond_tokens,
            query_pos=qpos_bqc,
            key_padding_mask=cond_pad_mask,
        )
        query_feat = q_bqc.permute(1, 0, 2)

        return tm.decoder(
            inputs_embeds=query_feat,
            multi_stage_positional_embeddings=multi_stage_pos_embeds,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_pos,
            feature_size_list=size_list,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

    def forward(self, pixel_values_m2f, input_ids, attention_mask, box=None):
        # Full text token sequence (no pooling).
        text_out    = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        text_seq    = text_out.last_hidden_state          # (B, T, text_hidden)
        text_tokens = self.text_proj(text_seq)            # (B, T, C)
        text_pad_mask = (attention_mask == 0)             # True = pad

        # Optional bbox conditioning. SAM corner tokens kept as a sequence.
        if box is not None and box.dim() == 2:
            box = box[:, None, :]                         # (B, 4) -> (B, 1, 4)
        if random.random() < self.bbx_random:
            box = None

        B = pixel_values_m2f.shape[0]
        if box is not None:
            with torch.no_grad():
                sparse_emb, _ = self.prompt_encoder(
                    input_points=None,
                    input_labels=None,
                    input_boxes=box,
                    input_masks=None,
                )
            # (B, num_boxes, 2, sam_hidden) -> (B, num_boxes*2, sam_hidden)
            # Typical call site uses num_boxes=1, so K=2.
            sparse_emb = sparse_emb.flatten(1, 2)
            box_tokens = self.box_proj(sparse_emb)        # (B, 2*num_boxes, C)
        else:
            # Two learnable no-box tokens, broadcast over batch. K=2,
            # matches the single-box (num_boxes=1) case above; if you
            # pass num_boxes>1, the two branches have different K.
            box_tokens = self.no_box_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 2, C)

        cond_tokens, cond_pad_mask = self._build_cond_tokens(
            text_tokens, text_pad_mask, box_tokens,
        )

        # Mask2Former path: native resolution, ImageNet normalization.
        encoder_out   = self.m2f_encoder(pixel_values_m2f)
        swin_features = list(encoder_out.feature_maps)

        pixel_dec_out        = self.pixel_decoder(swin_features)
        mask_features        = pixel_dec_out.mask_features
        multi_scale_features = list(pixel_dec_out.multi_scale_features)

        # Conditioned transformer decoder.
        tm_out       = self._run_transformer_with_cond(
            multi_scale_features, mask_features, cond_tokens, cond_pad_mask,
        )
        query_feats  = tm_out.last_hidden_state           # (B, Q, hidden)
        masks_logits = tm_out.masks_queries_logits[-1]    # (B, Q, h, w)

        class_logits = self.class_head(query_feats)       # (B, Q, 2)

        class_probs  = F.softmax(class_logits, dim=-1)    # softmax over classes per query
        mask_probs   = masks_logits.sigmoid()             # bounded [0, 1]

        seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        segmentation_output = F.interpolate(
            seg_logits,
            size=pixel_values_m2f.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        return segmentation_output, box
