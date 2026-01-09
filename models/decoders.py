import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_queries):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)  # Learnable queries
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)
        ])

    def forward(self, memory):
        B = memory.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, d_model)

        for layer in self.layers:
            queries = layer(queries, memory)

        return queries  



class CustomDecoder(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super(CustomDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 7x7 → 14x14

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14x14 → 28x28

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28x28 → 56x56

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 56x56 → 224x224

            nn.Conv2d(32, num_classes, kernel_size=1)  # Final conv: per-pixel class scores
        )

    def forward(self, x):
        return self.decoder(x)  # (B, num_classes, H, W)