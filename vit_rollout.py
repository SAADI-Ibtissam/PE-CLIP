import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

# def rollout(attentions, discard_ratio, head_fusion):
#     result = torch.eye(attentions[0].size(-1))
#     with torch.no_grad():
#         for attention in attentions:
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=1)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=1)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=1)[0]
#             else:
#                 raise "Attention head fusion type Not supported"

#             # Drop the lowest attentions, but
#             # don't drop the class token
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0

#             I = torch.eye(attention_heads_fused.size(-1))
#             a = (attention_heads_fused + 1.0*I)/2
#             a = a / a.sum(dim=-1)

#             result = torch.matmul(a, result)

def rollout(attentions, discard_ratio, head_fusion):
    if len(attentions) == 0:
        raise ValueError("No attention maps were collected during the forward pass.")

    # Initialize the result as an identity matrix
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type not supported")

            # Add identity to keep the class token
            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)



    # Extract class token attention to patches
    print(f"Shape of result: {result.shape}")  # Debugging
    if result.dim() == 2:  # Handle 2D case
        mask = result[0, 1:]  # Skip the class token
    elif result.dim() == 3:  # Handle 3D case
        mask = result[0, 0, 1:]  # Skip the class token
    else:
        raise ValueError(f"Unexpected result dimensions: {result.dim()}")

    # Compute the grid dimensions
    num_tokens = mask.size(-1)
    width = int(num_tokens ** 0.5)
    height = (num_tokens + width - 1) // width  # Round up to the nearest integer

    # Pad the mask if necessary
    if num_tokens < height * width:
        padding = height * width - num_tokens
        mask = torch.cat([mask, torch.zeros(padding, device=mask.device)], dim=-1)

    # Reshape to grid and normalize
    mask = mask.reshape(height, width).cpu().numpy()
    mask = mask / mask.max()
    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='module.image_encoder.transformer.resblocks.11.attn', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        if isinstance(output, tuple):  # Check if output is a tuple
            if len(output) > 1:
                attention_map = output[1]  # Adjust index based on your model's output structure
                self.attentions.append(attention_map.detach().cpu())
            else:
                raise ValueError("Tuple output does not contain sufficient elements.")
        else:  # Handle cases where output is a single tensor
            self.attentions.append(output.detach().cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
