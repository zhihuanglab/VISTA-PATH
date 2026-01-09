import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder
import random

from .decoders import CustomDecoder

class CustomSegmentationModel(nn.Module):
    def __init__(self, base_model_name, d_model, nhead, num_layers, bbx_random, vision_tune = False):
        super(CustomSegmentationModel, self).__init__()
        self.base_model = CLIPModel.from_pretrained(base_model_name)
        
        # Unfreeze specific layers of PLIP for fine-tuning

        if vision_tune:
            for name, param in self.base_model.named_parameters():
                if "vision_model" in name:
                    param.requires_grad = True  # Unfreeze vision and text encoders
                else:
                    param.requires_grad = False

        else:
            for name, param in self.base_model.named_parameters():
                    param.requires_grad = False
        
        self.cross_attn_text = CrossAttentionLayer(d_model=d_model, nhead=nhead)
        self.cross_attn_bbx = CrossAttentionLayer(d_model=d_model, nhead=nhead)

        self.image_proj = self.base_model.visual_projection  # usually nn.Linear
        self.text_proj = self.base_model.text_projection 
        
        # Define a transformer encoder layer
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Define the segmentation decoder
        self.decoder = CustomDecoder(input_channels=d_model)

        # sam_model = sam_model_registry['vit_b'](checkpoint='/project/zhihuanglab/Peixian/Path_Seg/DL/MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth')
        # self.prompt_encoder = sam_model.prompt_encoder

        self.prompt_encoder = PromptEncoder(
            embed_dim=512,
            image_embedding_size=(7, 7),
            input_image_size=(224, 224),
            mask_in_chans=16,
        )
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        

        self.bbx_random = bbx_random

    def forward(self, pixel_values, input_ids, attention_mask, box):

        with torch.no_grad():
            # box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box.shape) == 2:
                box = box[:, None, :]  # (B, 1, 4)
            

            if random.random() < self.bbx_random:
                box = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )

        outputs = self.base_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        # Extract vision transformer tokens: (B, 50, 512)
        image_tokens = outputs.vision_model_output.last_hidden_state[:, 1:, :]  # drop CLS token -> (B, 49, 512)

        # print(f"image_tokens: {image_tokens.shape}")
        image_proj = self.image_proj(image_tokens)  # (B, 49, d_model)
        # print(f"image_proj: {image_proj.shape}")

        # Project text and global-average pool for fusion
        text_tokens = outputs.text_model_output.last_hidden_state  # (B, T, 512)
        text_proj = self.text_proj(text_tokens)  # (B, T, d_model)

        # print(f"text_tokens: {text_tokens.shape}")
        # print(f"text_proj: {text_proj.shape}")

        # Cross-attention: let each image patch attend to the text
        fused_tokens = self.cross_attn_text(query=image_proj, key=text_proj, value=text_proj)  # (B, 49, d_model)

        fused_tokens = self.cross_attn_bbx(query=fused_tokens, key=sparse_embeddings, value=sparse_embeddings)  # (B, 49, d_model)

    

        # Transformer encoder (optional post-fusion modeling)
        fused_tokens = self.transformer_encoder(fused_tokens)  # (B, 49, d_model)

        B, N, D = fused_tokens.shape
        h = w = int(N ** 0.5)
        fused_feat = fused_tokens.permute(0, 2, 1).reshape(B, D, h, w)  # (B, d_model, 7, 7)


        # Segmentation output
        segmentation_output = self.decoder(fused_feat)  # (B, num_classes, H, W)

        if box is not None and box.shape[1] == 1:
            box = box[:, 0, :]  # (B, 4)

        return segmentation_output, box


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        output = self.norm(query + self.dropout(attn_output))  # Add & Norm
        return output