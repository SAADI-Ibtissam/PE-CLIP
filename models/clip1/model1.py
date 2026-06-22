# from collections import OrderedDict
# from typing import Tuple, Union

# import numpy as np
# import torch
# import torch.nn.functional as F
# #from nncore.nn import build_norm_layer
# from torch import nn
# import math

# # class MultiScaleSpatialAttention(nn.Module):
# #     def __init__(self, dim, reduction_factor=8):
# #         """
# #         Multi-Scale Spatial Attention Module
# #         dim: Dimension of the input features (e.g., from vision encoder).
# #         reduction_factor: Factor for dimensionality reduction in attention layers.
# #         """
# #         super(MultiScaleSpatialAttention, self).__init__()
        
# #         # Attention layers for three scales
# #         self.attn_scale1 = nn.Sequential(
# #             nn.Linear(dim, dim // reduction_factor),  # Reduce dimensionality
# #             nn.ReLU(),
# #             nn.Linear(dim // reduction_factor, 1),  # Compute attention score
# #             nn.Sigmoid()  # Normalize between 0 and 1
# #         )
        
# #         self.attn_scale2 = nn.Sequential(
# #             nn.Linear(dim, dim // reduction_factor),
# #             nn.ReLU(),
# #             nn.Linear(dim // reduction_factor, 1),
# #             nn.Sigmoid()
# #         )
        
# #         self.attn_scale3 = nn.Sequential(
# #             nn.Linear(dim, dim // reduction_factor),
# #             nn.ReLU(),
# #             nn.Linear(dim // reduction_factor, 1),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         """
# #         Forward pass for multi-scale spatial attention.
# #         x: Input tensor (batch_size, seq_len, dim)
# #         Returns:
# #             x: Tensor with multi-scale spatial attention applied.
# #         """
# #         batch_size, seq_len, dim = x.shape

# #         # Scale 1: Original scale
# #         attn1 = self.attn_scale1(x)  # (batch_size, seq_len, 1)

# #         # Scale 2: Downscale (simulate coarser resolution using pooling)
# #         downscaled_x = F.avg_pool1d(x.transpose(1, 2), kernel_size=2).transpose(1, 2)  # Downscale spatial dim
# #         attn2 = self.attn_scale2(downscaled_x)  # Compute attention at coarser scale
# #         attn2 = F.interpolate(attn2.transpose(1, 2), size=seq_len, mode='linear').transpose(1, 2)  # Upscale to original size

# #         # Scale 3: Upscale (simulate finer resolution using interpolation)
# #         upscaled_x = F.interpolate(x.transpose(1, 2), scale_factor=2, mode='linear').transpose(1, 2)
# #         attn3 = self.attn_scale3(upscaled_x[:, :seq_len, :])  # Compute attention at finer scale

# #         # Combine multi-scale attention maps
# #         combined_attn = attn1 + attn2 + attn3  # Element-wise sum of attention maps
# #         combined_attn = combined_attn / combined_attn.sum(dim=1, keepdim=True)  # Normalize across spatial dimension

# #         # Apply combined attention to input
# #         x = x * combined_attn  # Element-wise scaling by attention
# #         return x
    

# # class AdapterWithMultiScaleAttention(nn.Module):
# #     def __init__(self, dim, reduction_factor=8):
# #         super(AdapterWithMultiScaleAttention, self).__init__()
        
# #         # Multi-Scale Spatial Attention
# #         self.multi_scale_attn = MultiScaleSpatialAttention(dim, reduction_factor)
        
# #         # Standard Adapter Layers
# #         self.down_proj = nn.Linear(dim, dim // reduction_factor)
# #         self.activation = nn.GELU()
# #         self.up_proj = nn.Linear(dim // reduction_factor, dim)

# #     def forward(self, x):
# #         # Apply Multi-Scale Spatial Attention
# #         x = self.multi_scale_attn(x)
        
# #         # Apply Down-Projection, Activation, and Up-Projection
# #         residual = x
# #         x = self.down_proj(x)
# #         x = self.activation(x)
# #         x = self.up_proj(x)
        
# #         # Add Residual Connection
# #         return x + residual



# # class SpatialAdapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=8, use_attention=False, kernel_size=3):
# #         """
# #         Spatial Adapter using depthwise 1D convolution.
# #         Args:
# #             dim: Input feature dimension.
# #             reduction_factor: Bottleneck reduction factor.
# #             use_attention: Apply spatial attention (optional).
# #             kernel_size: Size of the convolution kernel.
# #         """
# #         super(SpatialAdapter, self).__init__()
# #         self.use_attention = use_attention

# #         # Bottleneck layers
# #         self.down_proj = nn.Linear(dim, dim // reduction_factor)
# #         self.activation = nn.GELU()
# #         self.up_proj = nn.Linear(dim // reduction_factor, dim)

# #         # Depthwise 1D convolution
# #         self.conv = nn.Conv1d(in_channels=dim // reduction_factor, 
# #                               out_channels=dim // reduction_factor, 
# #                               kernel_size=kernel_size, 
# #                               stride=1, 
# #                               padding=kernel_size // 2,  # Maintain sequence length
# #                               groups=dim // reduction_factor)  # Depthwise

# #         # Optional attention mechanism
# #         if use_attention:
# #             self.attention = nn.Sequential(
# #                 nn.Conv1d(dim // reduction_factor, dim // reduction_factor, kernel_size=1),
# #                 nn.Sigmoid()
# #             )

# #     def forward(self, x):
# #         """
# #         Args:
# #             x: Input tensor of shape [batch_size, seq_len, dim].
# #         Returns:
# #             Refined spatial features.
# #         """
# #         residual = x  # Save input for residual connection
# #         b, n, dim = x.shape

# #         # Step 1: Down-projection
# #         x = self.activation(self.down_proj(x))  # Shape: [b, n, reduced_dim]

# #         # Step 2: Permute for Conv1D
# #         x = x.permute(0, 2, 1)  # Shape: [b, reduced_dim, n]

# #         # Step 3: Apply depthwise convolution
# #         x = self.conv(x)

# #         # Step 4: Apply attention if enabled
# #         if self.use_attention:
# #             attention_weights = self.attention(x)
# #             x = x * attention_weights

# #         # Step 5: Permute back to original format
# #         x = x.permute(0, 2, 1)  # Shape: [b, n, reduced_dim]

# #         # Step 6: Up-projection
# #         x = self.up_proj(x)  # Shape: [b, n, dim]

# #         # Step 7: Residual connection
# #         x = x + residual
# #         return x

# #Region Aware SA###
# class SA_Adapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, region_count=4):
#         super(SA_Adapter, self).__init__()
#         assert dim % region_count == 0, "dim must be divisible by region_count"
#         self.down_proj = nn.Linear(dim, dim // reduction_factor)
#         self.activation = nn.GELU()
#         self.up_proj = nn.Linear(dim // reduction_factor, dim)

#         # Region-Aware Spatial Refinement
#         self.region_weights = nn.Parameter(torch.ones(1, region_count, 1), requires_grad=True)
#         self.region_count = region_count

#     def forward(self, x):
#         """
#         Forward pass with region-aware spatial refinement.
#         x: Input token embeddings (batch_size, seq_len, dim).
#         """
#         batch_size, seq_len, dim = x.shape
#         region_dim = dim // self.region_count

#         # Step 1: Reshape embeddings into regions
#         x_reshaped = x.view(batch_size, seq_len, self.region_count, region_dim)

#         # Step 2: Apply region weights (broadcasting along seq_len and region_dim)
#         x_refined = x_reshaped * self.region_weights  # Correct broadcasting

#         # Step 3: Flatten back to original dimensions
#         x = x_refined.view(batch_size, seq_len, dim)

#         # Step 4: Standard SA adapter bottleneck
#         x1 = self.down_proj(x)
#         x1 = self.activation(x1)
#         x1 = self.up_proj(x1)

#         # Step 5: Add residual connection
#         #x = x + x1

#         return x1

# # class SAAdapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=8):
# #         """
# #         Improved Spatial Attention Adapter for finer-grained features.
# #         dim: Dimension of the input features (e.g., from vision encoder).
# #         reduction_factor: Factor for dimensionality reduction in attention layers.
# #         """
# #         super(SAAdapter, self).__init__()
        
# #         # Local spatial context module (learnable)
# #         self.local_context = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # Depthwise convolution
        
# #         # Finer-grained spatial attention module
# #         self.finer_attention = nn.Sequential(
# #             nn.Linear(dim, dim // reduction_factor),  # Reduce dimensionality
# #             nn.ReLU(),
# #             nn.Linear(dim // reduction_factor, 1),  # Compute attention score
# #             nn.Sigmoid()  # Normalize between 0 and 1
# #         )
        
# #         # Feature refinement module
# #         self.refinement = nn.Sequential(
# #             nn.Linear(dim, dim // reduction_factor),
# #             nn.ReLU(),
# #             nn.Linear(dim // reduction_factor, dim)
# #         )

# #     def forward(self, x):
# #         """
# #         Forward pass with finer-grained spatial attention.
# #         x: Input tensor (batch_size, seq_len, dim).
# #         Returns:
# #             x: Tensor with improved spatial attention applied.
# #         """
# #         batch_size, seq_len, dim = x.shape

# #         # Step 1: Apply local spatial context (focus on nearby regions)
# #         local_features = self.local_context(x.transpose(1, 2))  # Depthwise convolution
# #         local_features = local_features.transpose(1, 2)  # Restore original shape

# #         # Step 2: Compute finer-grained spatial attention
# #         attention_weights = self.finer_attention(local_features)  # (batch_size, seq_len, 1)
# #         attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)  # Normalize

# #         # Step 3: Apply spatial attention
# #         x = x * attention_weights  # Element-wise scaling

# #         # Step 4: Refine features
# #         refined_features = self.refinement(x)
# #         x = x + refined_features  # Add residual connection for refinement

# #         return x

# #this is SA adapter
# # class SA_Adapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=8):
# #         super().__init__()
# #         self.spatial_attention = nn.Sequential(
# #             nn.Linear(dim, dim // reduction_factor),
# #             nn.GELU(),
# #             nn.Linear(dim // reduction_factor, 1),
# #             nn.Sigmoid()  # Normalize attention scores to [0, 1]
# #         )
# #         self.down_proj = nn.Linear(dim, dim // reduction_factor)
# #         self.up_proj = nn.Linear(dim // reduction_factor, dim)
# #         self.activation = nn.GELU()

# #     def forward(self, x):
# #         residual = x
# #         # Compute spatial attention scores
# #         attention_scores = self.spatial_attention(x)  # (batch_size, seq_len, 1)
# #         x = x * attention_scores  # Weight features by spatial importance

# #         # Down-projection and processing
# #         x = self.down_proj(x)
# #         x = self.activation(x)
# #         x = self.up_proj(x)

# #         return x + residual  # Add residual connection

# #### Spatio adapter #####
# # class Spatio_Adapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=8):
# #         """
# #         Adapter with spatial attention for vision encoder.
# #         dim: Dimension of input feature maps.
# #         reduction_factor: Reduction factor for bottleneck in the adapter.
# #         """
# #         super(Spatio_Adapter, self).__init__()
        
# #         # Dimension reduction and expansion layers
# #         self.down_proj = nn.Linear(dim, dim // reduction_factor)  # Reducing dimension
# #         self.activation = nn.GELU()  # Applying non-linearity
# #         self.up_proj = nn.Linear(dim // reduction_factor, dim)  # Restoring original dimension
        
# #         # Spatial attention mechanism
# #         self.spatial_attn = nn.Sequential(
# #             nn.Conv2d(dim, 1, kernel_size=1, stride=1),  # Generate attention map (1-channel mask)
# #             nn.Sigmoid()  # Normalize attention weights to [0, 1]
# #         )

# #     def forward(self, x):
# #         """
# #         Forward pass with spatial attention.
# #         x: Input tensor (batch_size, seq_len/spatial_dim, dim)
# #         Returns:
# #             x: Output tensor with spatial attention applied.
# #         """
# #         # Residual connection
# #         residual = x
        
# #         # Spatial attention computation
# #         batch_size, seq_len, dim = x.shape  # Assuming input is (batch_size, seq_len, dim)
# #         spatial_x = x.transpose(1, 2).reshape(batch_size, dim, int(seq_len**0.5), int(seq_len**0.5))  # Reshape to (B, C, H, W)
# #         attn_map = self.spatial_attn(spatial_x)  # Compute spatial attention (B, 1, H, W)
# #         spatial_x = spatial_x * attn_map  # Apply attention weights
# #         x = spatial_x.reshape(batch_size, dim, seq_len).transpose(1, 2)  # Reshape back to (B, seq_len, dim)
        
# #         # Dimension reduction and restoration
# #         x1 = self.down_proj(x)  # Reducing dimension
# #         x1 = self.activation(x1)  # Applying GELU activation
# #         x1 = self.up_proj(x1)  # Restoring original dimension
        
# #         # Add residual connection
# #         x = residual + x1  # Add original input back (residual connection)
# #         return x
    


# #### Adapter Bottelneck #####
# class Adapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8):
#         super().__init__()
#         self.down_proj = nn.Linear(dim, dim // reduction_factor)  # Reducing dimension
#         self.activation = nn.GELU()  # Applying non-linearity
#         self.up_proj = nn.Linear(dim // reduction_factor, dim)  # Restoring the original dimension
        
#     def forward(self, x):
#         residual = x
#         x1 = self.down_proj(x)  # Reducing dimension
#         x1 = self.activation(x1)  # Applying ReLU activation
#         x1 = self.up_proj(x1)
#         x = residual + x1  # Restoring dimension
#         return x   # Adding the original input back (residual connection)
    
# #textawareadapter####
# # class ContextualAttentionAdapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=8):
# #         super(ContextualAttentionAdapter, self).__init__()
# #         self.down_proj = nn.Linear(dim, dim // reduction_factor)
# #         self.activation = nn.GELU()
# #         self.up_proj = nn.Linear(dim // reduction_factor, dim)

# #         # Attention mechanism
# #         self.contextual_attention = nn.Linear(dim, 1)  # Scalar attention scores

# #     def forward(self, x):
# #         residual = x
# #         # Compute attention scores
# #         attention_scores = torch.softmax(self.contextual_attention(x), dim=1)  # (batch_size, seq_len, 1)
# #         x = x * attention_scores  # Apply attention to emphasize important features

# #         # Down-projection -> Activation -> Up-projection
# #         x1 = self.down_proj(x)
# #         x1 = self.activation(x1)
# #         x1 = self.up_proj(x1)

# #         return x1 + residual  # Residual connection


    
# # class RA_Adapter(nn.Module):
# #     def __init__(self,
# #                  d_model=768,
# #                  bottleneck=1024,
# #                 ):
# #         super().__init__()
# #         self.n_embd = d_model
# #         self.up_size = bottleneck
# #         self.up_proj = nn.Linear(self.n_embd, self.up_size)

# #         with torch.no_grad():
# #             nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
# #             torch.nn.init.uniform_(self.up_proj.bias)


# #     def forward(self, x):
# #         up = self.up_proj(x)
# #         return up


# # class TemporalAdapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=4, rnn_type='GRU'):
# #         super(TemporalAdapter, self).__init__()
        
# #         self.down_proj = nn.Linear(dim, dim // reduction_factor)
# #         self.activation = nn.GELU()

# #         # Recurrent layer
# #         if rnn_type == 'LSTM':
# #             self.rnn = nn.LSTM(dim // reduction_factor, dim // reduction_factor, batch_first=True)
# #         elif rnn_type == 'GRU':
# #             self.rnn = nn.GRU(dim // reduction_factor, dim // reduction_factor, batch_first=True)
# #         elif rnn_type == 'RNN':
# #             self.rnn = nn.RNN(dim // reduction_factor, dim // reduction_factor, batch_first=True)
# #         else:
# #             raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
        
# #         self.up_proj = nn.Linear(dim // reduction_factor, dim)

# #     def forward(self, x):
# #         # x shape: (batch_size, seq_len, dim)
     
# #         # Project down the dimensionality
# #         x1 = self.down_proj(x)
    
# #         # Apply the recurrent layer (RNN, LSTM, or GRU)
# #         x1, _ = self.rnn(x1)  # Ignore the hidden state output
# #         x1 = self.activation(x1)
# #         # Project back up to the original dimensionality
# #         x1 = self.up_proj(x1)
# #         # Add the residual connection
# #         x = x + x1
        
# #         return x

# # class TemporalAdapter(nn.Module):
   

# #     def __init__(self,
# #                  dims,
# #                  ratio=4,
# #                  p=0.1,
# #                  rank=32):
# #         super(TemporalAdapter, self).__init__()

# #         self._dims = dims
# #         self._ratio = ratio
# #         self._p = p
# #         self._h_dims = int(dims * ratio)
        
# #         self.first_linear_mapping_downprojection = nn.Linear(dims, rank)
# #         self.first_linear_mapping_upprojection = nn.Linear(rank, self._h_dims)
# #         self.first_act_layer = nn.ReLU()
# #         self.first_norm_layer = nn.LayerNorm(dims)
# #         self.second_linear_mapping_downprojection = nn.Linear(self._h_dims, rank)
# #         kernel_size = 3
# #         self.rnn = nn.RNN(rank, rank, batch_first=True)
# #         self.scale = nn.Linear(dims, 1)

# #         self.second_linear_mapping_upprojection = nn.Linear(rank, self._dims)
# #         self.second_norm_layer = nn.LayerNorm(dims)

# #         torch.nn.init.normal_(self.rnn.weight_ih_l0.data, 0.0, 1e-3)
# #         torch.nn.init.normal_(self.rnn.bias_ih_l0.data, 0.0, 1e-3)
# #         torch.nn.init.normal_(self.rnn.weight_hh_l0.data, 0.0, 1e-3)
# #         torch.nn.init.normal_(self.rnn.bias_hh_l0.data, 0.0, 1e-3)

# #         self.gelu = nn.GELU()

# #     def __repr__(self):
# #         return '{}(dims={}, ratio={}, p={})'.format(self.__class__.__name__,
# #                                                     self._dims, self._ratio,
# #                                                     self._p)


# #     def forward(self, x):
# #         x_original = x
# #         dynamic_scale = F.relu(self.scale(x)) 
# #         output = self.first_linear_mapping_downprojection(x)
# #         output = self.first_act_layer(output)
# #         output = self.first_linear_mapping_upprojection(output)
# #         #output = self.first_norm_layer(output)
# #         output = self.second_linear_mapping_downprojection(output)
# #         output = self.rnn(output)[0]
# #         output = self.gelu(output)
# #         output = self.second_linear_mapping_upprojection(output)
# #         #output = self.second_norm_layer(output)
# #         output = output * dynamic_scale

# #         # output = self.gelu(self.transformer_linear(output))

# #         output = x_original + output
# #         return output


# # class TemporalAdapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=8, rnn_type='GRU', adapter_layernorm_option="in", dropout=0.1):
# #         """
# #         dim: Dimension of the input tokens (from CLIP vision encoder).
# #         reduction_factor: Reduction factor for the bottleneck in the adapter.
# #         rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
# #         adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
# #         dropout: Dropout rate for regularization.
# #         """
# #         super(TemporalAdapter, self).__init__()

# #         self.n_embd = dim
# #         self.down_size = dim // reduction_factor
# #         self.adapter_layernorm_option = adapter_layernorm_option

# #         # LayerNorm setup (optional, based on adapter_layernorm_option)
# #         self.adapter_layer_norm_before = None
# #         if adapter_layernorm_option in {"in", "out"}:
# #             self.adapter_layer_norm_before = nn.LayerNorm(dim)

# #         # Dynamic scaling mechanism
# #         self.scale = nn.Linear(dim, 1)  # Optionally switch to Sigmoid for stability

# #         # Down-projection to reduce dimensionality
# #         self.down_proj = nn.Linear(dim, self.down_size)
# #         self.activation = nn.GELU()

# #         # Squeeze-and-Excitation Block (SE Block for Channel-Wise Attention)
# #         self.channel_attention = nn.Sequential(
# #             nn.AdaptiveAvgPool1d(1),  # Global average pooling (across temporal dimension)
# #             nn.Conv1d(self.down_size, self.down_size // 4, kernel_size=1, bias=False),  # Squeeze
# #             nn.ReLU(),
# #             nn.Conv1d(self.down_size // 4, self.down_size, kernel_size=1, bias=False),  # Excitation
# #             nn.Sigmoid()
# #         )

# #         # Recurrent layer to capture temporal dynamics
# #         if rnn_type == 'LSTM':
# #             self.rnn = nn.LSTM(self.down_size, self.down_size, batch_first=True)
# #         elif rnn_type == 'GRU':
# #             self.rnn = nn.GRU(self.down_size, self.down_size, batch_first=True)
# #         elif rnn_type == 'RNN':
# #             self.rnn = nn.RNN(self.down_size, self.down_size, batch_first=True)

# #         # Up-projection to restore original dimensionality
# #         self.up_proj = nn.Linear(self.down_size, dim)

# #         # Dropout for regularization
# #         self.dropout = nn.Dropout(dropout)

# #         # Initialize weights
# #         with torch.no_grad():
# #             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
# #             nn.init.zeros_(self.down_proj.bias)
# #             nn.init.zeros_(self.up_proj.bias)
# #             nn.init.zeros_(self.scale.bias)
# #             nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
# #             nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

# #     def forward(self, x):
# #         """
# #         Forward pass for the Temporal Adapter with SE block and dynamic scaling.
# #         x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
# #         Returns:
# #             - x: Adapted vision tokens with internal prompt concatenated.
# #         """
# #         # Save residual for skip connection
# #         residual = x

# #         # Step 1: Apply LayerNorm if set to 'in'
# #         if self.adapter_layernorm_option == 'in':
# #             x = self.adapter_layer_norm_before(x)

# #         # Step 2: Compute dynamic scaling factors for each token
# #         dynamic_scale = torch.sigmoid(self.scale(x))  # (batch_size, seq_len, 1)

# #         # Step 3: Down-projection (reduce dimensionality)
# #         down = self.down_proj(x)

# #         # Step 4: SE Block (Channel-wise Attention)
# #         down = down.permute(0, 2, 1)  # Switch to (batch_size, dim, seq_len) for SE block
# #         se_weights = self.channel_attention(down)  # Compute SE attention
# #         down = down * se_weights  # Apply SE attention
# #         down = down.permute(0, 2, 1)  # Back to (batch_size, seq_len, dim)

# #         # Step 5: Apply recurrent network (GRU/LSTM/RNN) to capture temporal dynamics
# #         down, _ = self.rnn(down)  # Ignore hidden states
# #         down = self.activation(down)
# #         down = self.dropout(down)

# #         # Step 6: Up-projection (restore original dimensionality)
# #         up = self.up_proj(down)
# #         #up = F.gelu(up)  # Optional: Add non-linearity after up-projection

# #         # Step 7: Apply dynamic scaling to the tokens
# #         up = up * dynamic_scale

# #         if self.adapter_layernorm_option == 'out':
# #             up = self.adapter_layer_norm_before(up)

# #         # Step 8: Residual connection
# #         x = up + residual

# #         return x




# class TemporalAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, rnn_type='GRU', scales=[1, 2, 4], adapter_layernorm_option="in", dropout=0.1):
#         super(TemporalAdapter, self).__init__()

#         self.n_embd = dim
#         self.down_size = dim // reduction_factor
#         self.adapter_layernorm_option = adapter_layernorm_option

#         # LayerNorm setup
#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option in ["in", "out"]:
#             self.adapter_layer_norm_before = nn.LayerNorm(dim)

#         # Dynamic scaling mechanism
#         self.scale = nn.Linear(dim, 1)

#         # Down-projection
#         self.down_proj = nn.Linear(dim, self.down_size)
#         self.activation = nn.GELU()

#         # Multi-scale RNNs
#         self.rnn_layers = nn.ModuleList()
#         for scale in scales:
#             if rnn_type == 'LSTM':
#                 self.rnn_layers.append(nn.LSTM(self.down_size, self.down_size, num_layers=scale, batch_first=True))
#             elif rnn_type == 'GRU':
#                 self.rnn_layers.append(nn.GRU(self.down_size, self.down_size, num_layers=scale, batch_first=True))
#             elif rnn_type == 'RNN':
#                 self.rnn_layers.append(nn.RNN(self.down_size, self.down_size, num_layers=scale, batch_first=True))

#         # Aggregation projection
#         self.aggregation_proj = nn.Linear(len(scales) * self.down_size, self.down_size)

#         # Up-projection
#         self.up_proj = nn.Linear(self.down_size, dim)

#         # Dropout
#         self.dropout = nn.Dropout(dropout)

#         # Initialization
#         nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.down_proj.bias)
#         nn.init.zeros_(self.up_proj.weight)
#         nn.init.zeros_(self.up_proj.bias)
#         nn.init.constant_(self.scale.weight, 0.0)
#         nn.init.zeros_(self.scale.bias)

#     def forward(self, x, add_residual=True, residual=None):
#         residual = x

#         # Step 1: LayerNorm (in)
#         if self.adapter_layernorm_option == "in":
#             x = self.adapter_layer_norm_before(x)

#         # Step 2: Dynamic scaling
#         dynamic_scale = torch.sigmoid(self.scale(x))  # (batch_size, seq_len, 1)

#         # Step 3: Down-projection
#         down = self.activation(self.down_proj(x))

#         # Step 4: Multi-scale RNNs
#         multi_scale_outputs = []
#         for rnn_layer in self.rnn_layers:
#             rnn_output, _ = rnn_layer(down)
#             multi_scale_outputs.append(rnn_output)

#         # Step 5: Aggregate multi-scale outputs
#         aggregated_features = torch.cat(multi_scale_outputs, dim=-1)
#         aggregated_features = self.aggregation_proj(aggregated_features)
#         aggregated_features = self.activation(aggregated_features)
#         aggregated_features = self.dropout(aggregated_features)

#         # Step 6: Up-projection
#         up = self.up_proj(aggregated_features)

#         # Step 7: Apply dynamic scaling
#         up = up * dynamic_scale

#         # Step 8: Residual connection
#         if add_residual:
#             x = up + residual
#         else:
#             x = up

#         # Step 9: LayerNorm (out)
#         if self.adapter_layernorm_option == "out":
#             x = self.adapter_layer_norm_before(x)

#         return x
# #### Multiscale GRU ######
# # class TemporalAdapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=8, rnn_type='GRU', scales=[1, 2, 4], adapter_layernorm_option="in", dropout=0.1):
# #         """
# #         dim: Dimension of the input tokens (from CLIP vision encoder).
# #         reduction_factor: Reduction factor for the bottleneck in the adapter.
# #         rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
# #         scales: List of temporal scales for multi-scale feature aggregation.
# #         adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
# #         dropout: Dropout rate for regularization.
# #         """
# #         super(TemporalAdapter, self).__init__()

# #         self.n_embd = dim
# #         self.down_size = dim // reduction_factor
# #         self.adapter_layernorm_option = adapter_layernorm_option

# #         # LayerNorm setup (optional, based on adapter_layernorm_option)
# #         self.adapter_layer_norm_before = None
# #         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
# #             self.adapter_layer_norm_before = nn.LayerNorm(dim)

# #         # Dynamic scaling mechanism
# #         self.scale = nn.Linear(dim, 1)

# #         # Down-projection to reduce dimensionality
# #         self.down_proj = nn.Linear(dim, self.down_size)
# #         self.activation = nn.GELU()

# #         # Multi-scale recurrent layers
# #         self.rnn_layers = nn.ModuleList()
# #         for scale in scales:
# #             if rnn_type == 'LSTM':
# #                 self.rnn_layers.append(nn.LSTM(self.down_size, self.down_size, num_layers=scale, batch_first=True))
# #             elif rnn_type == 'GRU':
# #                 self.rnn_layers.append(nn.GRU(self.down_size, self.down_size, num_layers=scale, batch_first=True))
# #             elif rnn_type == 'RNN':
# #                 self.rnn_layers.append(nn.RNN(self.down_size, self.down_size, num_layers=scale, batch_first=True))

# #         # Up-projection to restore original dimensionality
# #         self.up_proj = nn.Linear(self.down_size, dim)

# #         # Dropout for regularization
# #         self.dropout = nn.Dropout(dropout)

# #         # Initialize weights for up-projection and dynamic scale (optional)
# #         with torch.no_grad():
# #             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
# #             nn.init.zeros_(self.up_proj.weight)
# #             nn.init.zeros_(self.down_proj.bias)
# #             nn.init.zeros_(self.up_proj.bias)
# #             nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
# #             nn.init.zeros_(self.scale.bias)
# #             nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
# #             nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

# #     def forward(self, x, add_residual=True, residual=None):
# #         """
# #         Forward pass for multi-scale temporal adapter.
# #         x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
# #         Returns:
# #             - x: Adapted vision tokens with multi-scale feature aggregation.
# #         """
# #         # Save residual input if add_residual is True
# #         residual = x

# #         # Step 1: Apply LayerNorm if set to 'in'
# #         if self.adapter_layernorm_option == 'in':
# #             x = self.adapter_layer_norm_before(x)

# #         # Step 2: Compute dynamic scaling factors for each token
# #         dynamic_scale = torch.sigmoid(self.scale(x))  # (batch_size, seq_len, 1)

# #         # Step 3: Down-projection (reduce dimensionality)
# #         down = self.down_proj(x)

# #         # Step 4: Apply multi-scale recurrent layers
# #         multi_scale_outputs = []
# #         for rnn_layer in self.rnn_layers:
# #             rnn_output, _ = rnn_layer(down)
# #             multi_scale_outputs.append(rnn_output)

# #         # Step 5: Aggregate multi-scale outputs (e.g., summation or concatenation)
# #         aggregated_features = torch.stack(multi_scale_outputs, dim=0).mean(dim=0)  # Averaging across scales
# #         aggregated_features = self.activation(aggregated_features)
# #         aggregated_features = self.dropout(aggregated_features)

# #         # Step 6: Up-projection (restore original dimensionality)
# #         up = self.up_proj(aggregated_features)

# #         # Step 7: Apply dynamic scaling to the tokens
# #         up = up * dynamic_scale
# #         if self.adapter_layernorm_option == 'out':
# #             up = self.adapter_layer_norm_before(up)

# #         if add_residual:
# #             x = up + residual
# #         else:
# #             x = up

# #         return x

# # ###TDA based on GRU #### 
# # class TemporalAdapter(nn.Module):
# #     def __init__(self, dim, reduction_factor=4, rnn_type='GRU', prompt_dim=128, adapter_layernorm_option="in", dropout=0.1):
# #         """
# #         dim: Dimension of the input tokens (from CLIP vision encoder).
# #         reduction_factor: Reduction factor for the bottleneck in the adapter.
# #         rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
# #         prompt_dim: Dimensionality for the internal prompt.
# #         adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
# #         dropout: Dropout rate for regularization.
# #         """
# #         super(TemporalAdapter, self).__init__()

# #         self.n_embd = dim
# #         self.down_size = dim // reduction_factor
# #         self.adapter_layernorm_option = adapter_layernorm_option

# #         # LayerNorm setup (optional, based on adapter_layernorm_option)
# #         self.adapter_layer_norm_before = None
# #         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
# #             self.adapter_layer_norm_before = nn.LayerNorm(dim)

# #         # Dynamic scaling mechanism
# #         self.scale = nn.Linear(dim, 1)
    
# #         #self.gate = nn.Linear(dim, 1)

# #         # Down-projection to reduce dimensionality
# #         self.down_proj = nn.Linear(dim, self.down_size)
# #         self.activation = nn.GELU()
# #         #self.second_norm_layer = build_norm_layer('drop', p=0.1)

# #         # Recurrent layer to capture temporal dynamics (LSTM/GRU/RNN)
# #         if rnn_type == 'LSTM':
# #             self.rnn = nn.LSTM(self.down_size, self.down_size, batch_first=True)
# #         elif rnn_type == 'GRU':
# #             self.rnn = nn.GRU(self.down_size, self.down_size, batch_first=True)
# #         elif rnn_type == 'RNN':
# #             self.rnn = nn.RNN(self.down_size, self.down_size, batch_first=True)

# #         # Attention layer over RNN outputs
# #         #self.attention_layer = nn.Linear(self.down_size, 1)  # Attention weights for each time step

# #         # Up-projection to restore original dimensionality
# #         self.up_proj = nn.Linear(self.down_size , dim)

# #         # Dropout for regularization
# #         #self.dropout = nn.Dropout(dropout)

# #         # Initialize weights for up-projection and dynamic scale (optional)
# #         with torch.no_grad():
# #             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
# #             nn.init.zeros_(self.up_proj.weight)
# #             nn.init.zeros_(self.down_proj.bias)
# #             nn.init.zeros_(self.up_proj.bias)
# #             nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
# #             nn.init.zeros_(self.scale.bias)
# #             nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
# #             nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

# #     def forward(self, x, add_residual=True, residual=None):
# #         """
# #         Forward pass for dynamic temporal adapter with internal prompt generation.
# #         x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
# #         Returns:
# #             - x: Adapted vision tokens with internal prompt concatenated.
# #         """
# #         # Save residual input if add_residual is True
# #         residual = x 

# #         # Step 1: Apply LayerNorm if set to 'in'
# #         if self.adapter_layernorm_option == 'in':
# #             x = self.adapter_layer_norm_before(x)

# #         # Step 2: Compute dynamic scaling factors for each token
# #         #dynamic_scale = F.relu(self.scale(x))  # (batch_size, seq_len, 1)
# #         dynamic_scale = torch.sigmoid(self.scale(x))

# #         #gate = torch.sigmoid(self.gate(x))


# #         # Step 3: Down-projection (reduce dimensionality)
# #         down = self.down_proj(x)
        
# #         # Step 4: Apply recurrent network (LSTM/GRU/RNN) to capture temporal dynamics
# #         down, _ = self.rnn(down)  # Ignore hidden states
# #         #down = F.dropout(down, p=0.15, training=self.training)
# #         down = self.activation(down)
# #         #down = self.dropout(down)
# #         #down = nn.functional.dropout(down, p=self.dropout, training=self.training)
# #         # Step 5: Up-projection (restore original dimensionality)
# #         up = self.up_proj(down)
# #         #up = self.second_norm_layer(up)

# #         # Step 6: Apply dynamic scaling to the tokens
# #         up = up * dynamic_scale
# #         #up = up * gate
# #         if self.adapter_layernorm_option == 'out':
# #             up = self.adapter_layer_norm_before(up)

# #         if add_residual:
# #             x= up + residual

# #         else:
# #             x=up
    
# #         return x  

# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1):
#         super().__init__()

#         # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
#         self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = None
#         self.stride = stride

#         if stride > 1 or inplanes != planes * Bottleneck.expansion:
#             # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
#             self.downsample = nn.Sequential(OrderedDict([
#                 ("-1", nn.AvgPool2d(stride)),
#                 ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
#                 ("1", nn.BatchNorm2d(planes * self.expansion))
#             ]))

#     def forward(self, x: torch.Tensor):
#         identity = x

#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.avgpool(out)
#         out = self.bn3(self.conv3(out))

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)
#         return out


# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads

#     def forward(self, x):
#         x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
#         x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )

#         return x[0]


# class ModifiedResNet(nn.Module):
#     """
#     A ResNet class that is similar to torchvision's but contains the following changes:
#     - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
#     - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
#     - The final pooling layer is a QKV attention instead of an average pool
#     """

#     def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
#         super().__init__()
#         self.output_dim = output_dim
#         self.input_resolution = input_resolution

#         # the 3-layer stem
#         self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width // 2)
#         self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(width // 2)
#         self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(width)
#         self.avgpool = nn.AvgPool2d(2)
#         self.relu = nn.ReLU(inplace=True)

#         # residual layers
#         self._inplanes = width  # this is a *mutable* variable used during construction
#         self.layer1 = self._make_layer(width, layers[0])
#         self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

#         embed_dim = width * 32  # the ResNet feature dimension
#         self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

#     def _make_layer(self, planes, blocks, stride=1):
#         layers = [Bottleneck(self._inplanes, planes, stride)]

#         self._inplanes = planes * Bottleneck.expansion
#         for _ in range(1, blocks):
#             layers.append(Bottleneck(self._inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         def stem(x):
#             for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
#                 x = self.relu(bn(conv(x)))
#             x = self.avgpool(x)
#             return x

#         x = x.type(self.conv1.weight.dtype)
#         x = stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.attnpool(x)

#         return x


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)


# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)



# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, adapter_type: str = 'text'):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask

#         # Choose the correct adapter based on the adapter_type
#         if adapter_type == 'text':
#             self.adapter_pre_attn = None#Adapter(d_model)  # Text Adapter before attention
#             self.adapter_pre_mlp = Adapter(d_model)   # Text Adapter before MLP
#         elif adapter_type == 'vision':
#             self.adapter_pre_attn = None#TemporalAdapter(d_model)   # Temporal Adapter before attention for vision
#             self.adapter_pre_mlp = TemporalAdapter(d_model)   # Temporal Adapter before MLP for vision
#         else:
#             raise ValueError("adapter_type must be 'text' or 'vision'")

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x: torch.Tensor):
#     # Apply attention mechanism first, then apply adapter after
#         x = x + self.attention(self.ln_1(x))
#         x = self.adapter_pre_attn(x)
#         # if self.adapter_pre_attn is not None:
#         #  x = self.adapter_pre_attn(x)

#         # Apply MLP first, then apply adapter after
#         x = x + self.mlp(self.ln_2(x))
#         if self.adapter_pre_mlp is not None:
#          x = self.adapter_pre_mlp(x)
    
#         return x

# class ResidualAttentionBlock_IVLP(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
#                  text_layer=False, i=0, design_details=None):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.text_layer = text_layer
#         self.attn_mask = attn_mask
     
#         if i != 0:
#             self.add_prompt = add_prompt
#             if self.add_prompt:
#                 if self.text_layer:
#                     self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
#                     ctx_vectors = torch.empty(self.n_ctx_text, d_model)
#                 else:
#                     self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
#                     ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
#                 nn.init.normal_(ctx_vectors, std=0.02)
#                 self.VPT_shallow = nn.Parameter(ctx_vectors)
#         else:
#             self.add_prompt = False

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x: torch.Tensor):
#         if self.add_prompt:
#             if not self.text_layer:
#                 prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
#                 visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
#                 x = torch.cat([prefix, visual_context], dim=0)
#             else:
#                 prefix = x[:1, :, :]
#                 suffix = x[1 + self.n_ctx_text:, :, :]
#                 textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
#                 x = torch.cat([prefix, textual_context, suffix], dim=0)

#         x = x + self.attention(self.ln_1(x))
      
#         x = x + self.mlp(self.ln_2(x))
        
#         return x


# class ResidualAttentionBlock_MaPLe(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
#                  text_layer=False, i=0, adapter_type: str = 'text', shared_adapter=None):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.text_layer = text_layer
#         self.attn_mask = attn_mask
#         #shared_adapters = Adapter(d_model)
        

#        # Choose the correct adapter based on the adapter_type
#         if adapter_type == 'text':
#             self.adapter_pre_attn =None#shared_adapters#ContextualAttentionAdapter(d_model)#shared_adapters#Adapter(d_model)  # Text Adapter before attention
#             self.adapter_pre_mlp = None#shared_adapters#Adapter(d_model)   # Text Adapter before MLP
#             #self.adapter_pre_text = Adapter(d_model)
#         elif adapter_type == 'vision':
#             self.adapter_pre_attn = None#SA_Adapter(d_model) #Adapter(d_model)#None#SAAdapter(d_model)#shared_adapters#Spatio_Adapter(d_model)#shared_adapters#TemporalAdapter(d_model)   # Temporal Adapter before attention for vision
#             self.adapter_pre_mlp = TemporalAdapter(d_model)   # Temporal Adapter before MLP for vision
#             #self.adapter_pre_text = None
#             #self.adapter_pre_mlp = TemporalAdapter(d_model)   # Temporal Adapter before MLP for vision
#         else:
#             raise ValueError("adapter_type must be 'text' or 'vision'")
        
#         #self.shared_adapter = shared_adapter

#         self.compound_prompt_nctx = design_details['maple_length']
#         self.first_layer = (i == 0)
#         self.non_linear_func = nn.GELU()

       
#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    
#     def forward(self, inputs):
#         x = inputs[0]
#         compound_prompts_deeper = inputs[1]
#         counter = inputs[2]
        
#         if not self.first_layer:
#             if len(compound_prompts_deeper) > 0:
#                 if not self.text_layer:  # Vision layer
#                     if not (counter > len(compound_prompts_deeper) - 1):
#                         prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
#                         visual_context = compound_prompts_deeper[counter]
#                         visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
#                         x = torch.cat([prefix, visual_context], dim=0)
#                         counter += 1
#                 else:  # Text layer
#                     if not (counter > len(compound_prompts_deeper) - 1):
#                         prefix = x[:1, :, :]
#                         suffix = x[1 + self.compound_prompt_nctx:, :, :]
#                         textual_context = compound_prompts_deeper[counter]
#                         textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
#                         x = torch.cat([prefix, textual_context, suffix], dim=0)
#                         counter += 1
    
#     # Apply attention mechanism first, then apply adapter after
#         # x = x + self.attention(self.ln_1(x))
#         # if self.adapter_pre_attn is not None:
#         #  x = self.adapter_pre_attn(x)
      
#         # # Apply MLP first, then apply adapter after
#         # x = x + self.mlp(self.ln_2(x))
#         # if self.adapter_pre_mlp is not None:
#         #  x = self.adapter_pre_mlp(x)
    
#         residual = x

#         # if self.adapter_pre_attn is not None:
#         #     x = self.adapter_pre_attn(x)
#         # x = x + self.attention(self.ln_1(x))
    
#         # if self.adapter_pre_mlp is not None:
#         #    x = self.adapter_pre_mlp(x)
          
#         # x = x + self.mlp(self.ln_2(x))   
       
#         #### best way#######
#         x_attn = self.attention(self.ln_1(x))  # Apply attention
#         if self.adapter_pre_attn is not None:
#             x_attn = self.adapter_pre_attn(x_attn)  # Apply adapter after attention
#         x = residual + x_attn  # Apply residual connection

#         # Apply MLP first, then the adapter, and finally apply the residual connection
#         x_residual = x 
#         x_mlp = self.mlp(self.ln_2(x))  # Apply MLP
#         if self.adapter_pre_mlp is not None:
#             x_mlp = self.adapter_pre_mlp(x_mlp)  # Apply adapter after MLP
#         x = x_residual +  x_mlp  # Apply residual connection
#         return [x, compound_prompts_deeper, counter]
#         ##############################
#         # if self.adapter_pre_mlp is not None:
#         #     x = self.adapter_pre_mlp(x)

#         # x_attn = self.attention(self.ln_1(x))  # Apply attention
#         # if self.adapter_pre_attn is not None:
#         #     x_attn = self.adapter_pre_attn(x_attn)
#         #   # Apply adapter after attention
#         # x = residual + x_attn  # Apply residual connection
#         # # Apply MLP first, then the adapter, and finally apply the residual connection
#         # x_residual = x 
#         # x_mlp = self.mlp(self.ln_2(x))  # Apply MLP
#         # if self.adapter_pre_text is not None:
#         #     x_shared = self.adapter_pre_text(x_mlp)  # Apply adapter after MLP
#         # else:
#         #     x_shared = x_mlp
#         # x = x_residual +  x_shared  # Apply residual connection

        



# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
#                  text_layer=False, design_details=None,  adapter_type: str = 'text', shared_adapter=None): #, shared_adapter=None
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         # Implements respective encoder blocks for a given design choice
#         current_trainer = design_details['trainer']
#         if current_trainer == 'IVLP' or current_trainer == 'VPT':
#             self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
#                                                                          text_layer, i,
#                                                                          design_details) if prompts_needed > i
#                                              else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
#                                                                               text_layer, i, design_details)
#                                              for i in range(layers)])
#         elif current_trainer == 'MaPLe':
#             self.resblocks = nn.Sequential(
#                 *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer=text_layer, i=i, adapter_type=adapter_type, shared_adapter=shared_adapter)
#                   for i in range(layers)]) #, shared_adapter=shared_adapter[i]
#         else:
#             # Corresponds to default CoOp or CoCoOp
#             assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
#             self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)


# class VisionTransformer(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
#                  output_dim: int, design_details, shared_adapter=None):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
#         if design_details["vision_depth"] == 0:
#             self.VPT_shallow = False
#         else:
#             self.VPT_shallow = True
#         if self.VPT_shallow:
#             # Add visual prompt tokens here
#             n_ctx = design_details["vision_ctx"]  # hyperparameter
#             ctx_vectors = torch.empty(n_ctx, width)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             self.VPT = nn.Parameter(ctx_vectors)
#             # self.VPT.half()
#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)
#         # hyper-parameter if need to add prompt embeddings inside to the input
#         # of transformer block or not:
#         self.prompt_till_layer_visual = design_details["vision_depth"]
#         self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
#                                        design_details=design_details, adapter_type='vision', shared_adapter=shared_adapter)

#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#     def forward(self, x: torch.Tensor):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)

#         # After positional embeddings, we will attach prompts with the model, remember only those
#         # are trainable parameters here in whole image encoder.
#         if self.VPT_shallow:
#             visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
#             x = torch.cat([x, visual_ctx], dim=1)
#         else:
#             assert self.prompt_till_layer_visual == 0

#         # Normal code as before
#         x = self.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         x = self.ln_post(x[:, 0, :])

#         if self.proj is not None:
#             x = x @ self.proj

#         return x


# class VisionTransformer_MaPLe(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
#                  design_details, shared_adapter=None):#, shared_adapter=None
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
#         self.VPT_shallow = True
#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)
#         # hyper-parameter if need to add prompt embeddings inside to the input
#         # of transformer block or not:
#         self.prompt_till_layer_visual = 0
#         self.transformer = Transformer(width, layers, heads, design_details=design_details, adapter_type='vision', shared_adapter=shared_adapter)#, shared_adapter=shared_adapter

#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#     def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)

#         # After positional embeddings, we will attach prompts with the model, remember only those
#         # are trainable parameters here in whole image encoder.
#         if self.VPT_shallow:
#             visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
#             x = torch.cat([x, visual_ctx], dim=1)
#         else:
#             assert self.prompt_till_layer_visual == 0

#         # Normal code as before
#         x = self.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         # Again combine the inputs, so nn.sequential can work
#         outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
#         x = outputs[0]
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         x = self.ln_post(x[:, 0, :])

#         if self.proj is not None:
#             x = x @ self.proj

#         return x


# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  vision_patch_size: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int,
#                  design_details
#                  ):
#         super().__init__()

#         self.context_length = context_length
#         trainer = design_details['trainer']

#         #self.shared_adapters = Adapter(embed_dim)
#         #     for i in range(transformer_layers)])
        
#         if isinstance(vision_layers, (tuple, list)):
#             vision_heads = vision_width * 32 // 64
#             self.visual = ModifiedResNet(
#                 layers=vision_layers,
#                 output_dim=embed_dim,
#                 heads=vision_heads,
#                 input_resolution=image_resolution,
#                 width=vision_width
#             )
#         else:
#             vision_heads = vision_width // 64
#             if trainer == "MaPLe":
#                 self.visual = VisionTransformer_MaPLe(
#                     input_resolution=image_resolution,
#                     patch_size=vision_patch_size,
#                     width=vision_width,
#                     layers=vision_layers,
#                     heads=vision_heads,
#                     output_dim=embed_dim,
#                     design_details=design_details
#                     #shared_adapter = self.shared_adapters

#                 )
#             else:
#                 self.visual = VisionTransformer(
#                     input_resolution=image_resolution,
#                     patch_size=vision_patch_size,
#                     width=vision_width,
#                     layers=vision_layers,
#                     heads=vision_heads,
#                     output_dim=embed_dim,
#                     design_details=design_details
#                 )
#         # hyper-parameter if need to add prompt embeddings inside to the input
#         # of transformer block or not:
#         prompt_till_layer_text = design_details['language_depth']
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask(),
#             prompts_needed=prompt_till_layer_text,
#             text_layer=True,
#             design_details=design_details,
#             adapter_type='text' 
#             #shared_adapter = self.shared_adapters
#         )

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)

#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.initialize_parameters()

#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)

#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)

#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         if self.text_projection is not None:
#             nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     @property
#     def dtype(self):
#         return self.visual.conv1.weight.dtype

#     def encode_image(self, image):
#         return self.visual(image.type(self.dtype))

#     def encode_text(self, text):
#         x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

#         x = x + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

#         return x

#     def forward(self, image, text):
#         image_features = self.encode_image(image)
#         text_features = self.encode_text(text)

#         # normalized features
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logit_scale * text_features @ image_features.t()

#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_text


# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""

#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()

#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()

#     model.apply(_convert_weights_to_fp16)


# def build_model(state_dict: dict, design_details):
#     vit = "visual.proj" in state_dict

#     if vit:
#         vision_width = state_dict["visual.conv1.weight"].shape[0]
#         vision_layers = len(
#             [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#         vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
#         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size
#     else:
#         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
#                         [1, 2, 3, 4]]
#         vision_layers = tuple(counts)
#         vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
#         output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#         vision_patch_size = None
#         assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
#         image_resolution = output_width * 32

#     embed_dim = state_dict["text_projection"].shape[1]
#     context_length = state_dict["positional_embedding"].shape[0]
#     vocab_size = state_dict["token_embedding.weight"].shape[0]
#     transformer_width = state_dict["ln_final.weight"].shape[0]
#     transformer_heads = transformer_width // 64
#     transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

#     model = CLIP(
#         embed_dim,
#         image_resolution, vision_layers, vision_width, vision_patch_size,
#         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
#     )

#     for key in ["input_resolution", "context_length", "vocab_size"]:
#         if key in state_dict:
#             del state_dict[key]

#     convert_weights(model)
#     try:
#         model.load_state_dict(state_dict, strict=False)
#     except:
#         missing_keys, _ = model.load_state_dict(state_dict, strict=False)
#         print('Weights not found for some missing keys: ', missing_keys)
#     return model.eval()


#####################################################Base Model###################################################################
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
#from nncore.nn import build_norm_layer
from .ConvLSTM import ConvLSTM
from torch import nn
from models.Temporal_Model import *
#from torch.nn.modules.rnn import ConvLSTM
import math



#### Adapter Bottelneck #####
class Adapter(nn.Module):
    def __init__(self, dim, reduction_factor=8):
        super().__init__()
        self.down_proj = nn.Linear(dim, dim // reduction_factor)  # Reducing dimension
        self.activation = nn.GELU()  # Applying non-linearity
        self.up_proj = nn.Linear(dim // reduction_factor, dim)  # Restoring the original dimension
        
    def forward(self, x):
        residual = x
        x1 = self.down_proj(x)  # Reducing dimension
        x1 = self.activation(x1)  # Applying ReLU activation
        x1 = self.up_proj(x1)
        x = residual + x1  # Restoring dimension
        return x   # Adding the original input back (residual connection)


# class RegionAwareAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, num_regions=4):
#         super().__init__()
#         self.num_regions = num_regions
#         self.region_attention = nn.Linear(dim, num_regions)  # Compute regional attention scores
#         self.down_proj = nn.Linear(dim, dim // reduction_factor)
#         self.activation = nn.GELU()
#         self.up_proj = nn.Linear(dim // reduction_factor, dim)

#     def forward(self, x):
#         residual = x  # Save input for residual connection
#         b, s, d = x.shape  # Batch, Sequence (Spatial), Dimension

#         # Step 1: Compute region-aware attention scores
#         region_scores = F.softmax(self.region_attention(x), dim=1)  # (B, S, num_regions)
        
#         # Step 2: Aggregate regional features
#         region_features = torch.einsum('bsd,bsr->brd', x, region_scores)  # (B, num_regions, D)
        
#         # Step 3: Bottleneck adapter transformation
#         region_features = self.down_proj(region_features)  # (B, num_regions, D // reduction_factor)
#         region_features = self.activation(region_features)
#         region_features = self.up_proj(region_features)  # (B, num_regions, D)
        
#         # Step 4: Re-project features back to spatial dimension
#         x_refined = torch.einsum('brd,bsr->bsd', region_features, region_scores)  # (B, S, D)
        
#         # Step 5: Residual connection
#         return x_refined + residual

    
### Spatial Adapter #######
# class SpatialAttentionAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8):
#         super().__init__()
#         # Spatial Attention Mechanism
#         self.spatial_attention = nn.Sequential(
#             nn.Linear(dim, dim // reduction_factor),
#             nn.GELU(),
#             nn.Linear(dim // reduction_factor, 1),
#             nn.Sigmoid()  # Normalize attention scores to [0, 1]
#         )
#         # Bottleneck Adapter
#         self.down_proj = nn.Linear(dim, dim // reduction_factor)
#         self.activation = nn.GELU()
#         self.up_proj = nn.Linear(dim // reduction_factor, dim)

#     def forward(self, x):
#         residual = x
#         # Compute Spatial Attention Scores
#         attention_scores = self.spatial_attention(x)  # (batch_size, seq_len, 1)
#         x = x * attention_scores  # Weight features by spatial importance

#         # Bottleneck Projection
#         x = self.down_proj(x)
#         x = self.activation(x)
#         x = self.up_proj(x)

#         return x + residual  # Residual Connection

    

####MS-GRU###########

# class TemporalAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, rnn_type='GRU', scales=[1, 2, 4], adapter_layernorm_option="in", dropout=0.1):
#         """
#         dim: Dimension of the input tokens (from CLIP vision encoder).
#         reduction_factor: Reduction factor for the bottleneck in the adapter.
#         rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
#         scales: List of temporal scales for multi-scale feature aggregation.
#         adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
#         dropout: Dropout rate for regularization.
#         """
#         super(TemporalAdapter, self).__init__()

#         self.n_embd = dim
#         self.down_size = dim // reduction_factor
#         self.adapter_layernorm_option = adapter_layernorm_option

#         # LayerNorm setup (optional, based on adapter_layernorm_option)
#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(dim)

#         # Dynamic scaling mechanism
#         self.scale = nn.Linear(dim, 1)

#         # Down-projection to reduce dimensionality
#         self.down_proj = nn.Linear(dim, self.down_size)
#         self.activation = nn.GELU()

#         # Multi-scale recurrent layers
#         self.rnn_layers = nn.ModuleList()
#         for scale in scales:
#             if rnn_type == 'LSTM':
#                 self.rnn_layers.append(nn.LSTM(self.down_size, self.down_size, num_layers=scale, batch_first=True))
#             elif rnn_type == 'GRU':
#                 self.rnn_layers.append(nn.GRU(self.down_size, self.down_size, num_layers=scale, batch_first=True))
#             elif rnn_type == 'RNN':
#                 self.rnn_layers.append(nn.RNN(self.down_size, self.down_size, num_layers=scale, batch_first=True))

#         #Aggregation projection
#         self.aggregation_proj = nn.Linear(len(scales) * self.down_size, self.down_size)

#         # Up-projection to restore original dimensionality
#         self.up_proj = nn.Linear(self.down_size, dim)

#         # Dropout for regularization
#         self.dropout = nn.Dropout(dropout)

#         # Initialize weights for up-projection and dynamic scale (optional)
#         with torch.no_grad():
#             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.up_proj.weight)
#             nn.init.zeros_(self.down_proj.bias)
#             nn.init.zeros_(self.up_proj.bias)
#             nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.scale.bias)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

#     def forward(self, x, add_residual=True, residual=None):
#         """
#         Forward pass for multi-scale temporal adapter.
#         x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
#         Returns:
#             - x: Adapted vision tokens with multi-scale feature aggregation.
#         """
#         # Save residual input if add_residual is True
#         residual = x

#         # Step 1: Apply LayerNorm if set to 'in'
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

#         # Step 2: Compute dynamic scaling factors for each token
#         dynamic_scale = torch.sigmoid(self.scale(x))  # (batch_size, seq_len, 1)

#         # Step 3: Down-projection (reduce dimensionality)
#         down = self.down_proj(x)

#         # Step 4: Apply multi-scale recurrent layers
#         multi_scale_outputs = []
#         for rnn_layer in self.rnn_layers:
#             rnn_output, _ = rnn_layer(down)
#             multi_scale_outputs.append(rnn_output)

#         # Step 5: Aggregate multi-scale outputs (e.g., summation or concatenation)
#         aggregated_features = torch.cat(multi_scale_outputs, dim=-1)
#         aggregated_features = self.aggregation_proj(aggregated_features)
#         aggregated_features = self.activation(aggregated_features)
#         aggregated_features = self.dropout(aggregated_features)

#         # Step 6: Up-projection (restore original dimensionality)
#         up = self.up_proj(aggregated_features)

#         # Step 7: Apply dynamic scaling to the tokens
#         up = up * dynamic_scale
#         if self.adapter_layernorm_option == 'out':
#             up = self.adapter_layer_norm_before(up)

#         if add_residual:
#             x = up + residual
#         else:
#             x = up

#         return x
    

# class TemporalAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, adapter_layernorm_option="in", dropout=0.1):
#         super(TemporalAdapter, self).__init__()

#         self.n_embd = dim
#         self.down_size = dim // reduction_factor

#         # LayerNorm setup
#         self.layer_norm = nn.LayerNorm(dim) if adapter_layernorm_option == "in" else None

#         # Temporal importance projection
#         self.temporal_weight_proj = nn.Linear(dim, 1)  # For relative temporal weighting
#         self.dynamic_scale_proj = nn.Linear(dim, 1)   # For absolute dynamic scaling

#         # Down-projection
#         self.down_proj = nn.Linear(dim * 2, self.down_size)  # Account for motion features
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(dropout)

#         # GRU for temporal modeling
#         self.rnn = nn.GRU(self.down_size, self.down_size, num_layers=1, batch_first=True)

#         # Up-projection
#         self.up_proj = nn.Linear(self.down_size, dim)

#         # Initialization
#         with torch.no_grad():
#             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#             nn.init.kaiming_uniform_(self.dynamic_scale_proj.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.down_proj.bias)
#             nn.init.zeros_(self.temporal_weight_proj.bias)
#             nn.init.zeros_(self.dynamic_scale_proj.bias)
#             for name, param in self.rnn.named_parameters():
#                 if "weight" in name:
#                     nn.init.kaiming_uniform_(param, a=math.sqrt(5))
#                 elif "bias" in name:
#                     nn.init.zeros_(param)

#     def forward(self, x):
#         residual = x

#         # Step 1: LayerNorm
#         if self.layer_norm:
#             x = self.layer_norm(x)

#         # Step 2: Frame differences (motion features)
#         frame_diff = x[:, 1:] - x[:, :-1]
#         frame_diff = torch.cat([frame_diff, frame_diff[:, -1:, :]], dim=1)  # Smooth padding
#         x_combined = torch.cat((x, frame_diff), dim=-1)

#         # Step 3: Compute combined scaling (dynamic scale + temporal weighting)
#         temporal_weights = torch.softmax(self.temporal_weight_proj(x), dim=1)  # Relative importance
#         dynamic_scale = torch.sigmoid(self.dynamic_scale_proj(x))              # Absolute importance
#         combined_scale = (dynamic_scale * temporal_weights) + 1e-6            # Stability

#         # Apply scaling to input features
#         weighted_x = x_combined * combined_scale

#         # Step 4: Down-projection
#         down = self.down_proj(weighted_x)

#         # Step 5: GRU processing
#         down, _ = self.rnn(down)
#         down = self.dropout(down)
#         down = self.activation(down)

#         # Step 6: Up-projection
#         up = self.up_proj(down)

#         # Step 7: Residual connection
#         if up.shape != residual.shape:
#             raise ValueError(f"Shape mismatch: up={up.shape}, residual={residual.shape}")
#         x = up + residual

#         return x


# ###Official___TDA based on GRU #### 
class TemporalAdapter(nn.Module):
    def __init__(self, dim, reduction_factor=8, rnn_type='GRU', prompt_dim=128, adapter_layernorm_option="in", dropout=0.1):
        """
        dim: Dimension of the input tokens (from CLIP vision encoder).
        reduction_factor: Reduction factor for the bottleneck in the adapter.
        rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
        prompt_dim: Dimensionality for the internal prompt.
        adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
        dropout: Dropout rate for regularization.
        """
        super(TemporalAdapter, self).__init__()

        self.n_embd = dim
        self.down_size = dim // reduction_factor
        self.adapter_layernorm_option = adapter_layernorm_option

        # LayerNorm setup (optional, based on adapter_layernorm_option)
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(dim)

        # Dynamic scaling mechanism
        self.scale = nn.Linear(dim, 1)
       

        # Down-projection to reduce dimensionality
        self.down_proj = nn.Linear(dim, self.down_size)
        self.activation = nn.GELU()
       
        # Recurrent layer to capture temporal dynamics (LSTM/GRU/RNN)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.down_size, self.down_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(self.down_size, self.down_size, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(self.down_size, self.down_size, batch_first=True)

      

        # Up-projection to restore original dimensionality
        self.up_proj = nn.Linear(self.down_size  , dim)

        # Dropout for regularization
        #self.dropout = nn.Dropout(dropout)

        # Initialize weights for up-projection and dynamic scale (optional)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
            nn.init.zeros_(self.scale.bias)
            nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
            nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

    def forward(self, x, add_residual=True, residual=None):
        """
        Forward pass for dynamic temporal adapter with internal prompt generation.
        x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
        Returns:
            - x: Adapted vision tokens with internal prompt concatenated.
        """
        # Save residual input if add_residual is True
        residual = x 

        # # Step 1: Apply LayerNorm if set to 'in'
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

      
        # Step 2: Compute dynamic scaling factors for each token
        dynamic_scale = F.relu(self.scale(x))  # (batch_size, seq_len, 1)
        #dynamic_scale = torch.sigmoid(self.scale(x))
        #weighted_x = x * dynamic_scale

        # Step 3: Down-projection (reduce dimensionality)
        down = self.down_proj(x)
        
        # Step 4: Apply recurrent network (LSTM/GRU/RNN) to capture temporal dynamics
        down, _ = self.rnn(down)  # Ignore hidden states
        #down = F.dropout(down, p=0.15, training=self.training)
        down = self.activation(down)
        #down = self.dropout(down)
        #down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        # Step 5: Up-projection (restore original dimensionality)
        up = self.up_proj(down)
        #up = self.second_norm_layer(up)
        
        #dynamic_scale = F.sigmoid(self.scale(up)) 
        # Step 6: Apply dynamic scaling to the tokens
        up = up * dynamic_scale
        #up = up * gate
        # if self.adapter_layernorm_option == 'out':
        #     up = self.adapter_layer_norm_before(up)
        if add_residual:
            x = up + residual

        else:
            x= up
    
        return x  

# ###TDA based on TransformerEncoder(1 layer)#### 
# class TemporalAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, num_heads=8, ff_multiplier=2, prompt_dim=128, adapter_layernorm_option="in", dropout=0.1):
#         """
#         dim: Dimension of the input tokens (from CLIP vision encoder).
#         reduction_factor: Reduction factor for the bottleneck in the adapter.
#         rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
#         prompt_dim: Dimensionality for the internal prompt.
#         adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
#         dropout: Dropout rate for regularization.
#         """
#         super(TemporalAdapter, self).__init__()

#         self.n_embd = dim
#         self.down_size = dim // reduction_factor
#         self.adapter_layernorm_option = adapter_layernorm_option

#         # LayerNorm setup (optional, based on adapter_layernorm_option)
#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(dim)

#         # Dynamic scaling mechanism
#         self.scale = nn.Linear(dim, 1)
       

#         # Down-projection to reduce dimensionality
#         self.down_proj = nn.Linear(dim, self.down_size)
#         self.activation = nn.GELU()
       
#         # Recurrent layer to capture temporal dynamics (LSTM/GRU/RNN)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.down_size,
#             nhead=num_heads,
#             dim_feedforward=self.down_size * ff_multiplier,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

      

#         # Up-projection to restore original dimensionality
#         self.up_proj = nn.Linear(self.down_size  , dim)

#         # Dropout for regularization
#         #self.dropout = nn.Dropout(dropout)

#         # Initialize weights for up-projection and dynamic scale (optional)
#         with torch.no_grad():
#             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.up_proj.weight)
#             nn.init.zeros_(self.down_proj.bias)
#             nn.init.zeros_(self.up_proj.bias)
#             nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.scale.bias)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

#     def forward(self, x, add_residual=True, residual=None):
#         """
#         Forward pass for dynamic temporal adapter with internal prompt generation.
#         x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
#         Returns:
#             - x: Adapted vision tokens with internal prompt concatenated.
#         """
#         # Save residual input if add_residual is True
#         residual = x 

#         # # Step 1: Apply LayerNorm if set to 'in'
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

      
#         # Step 2: Compute dynamic scaling factors for each token
#         dynamic_scale = F.relu(self.scale(x))  # (batch_size, seq_len, 1)
#         down = self.activation(self.down_proj(x))  # (B, T, D)
#         down = self.transformer(down)
#         up = self.up_proj(down)
#         up = up * dynamic_scale
   
#         if add_residual:
#             x = up + residual

#         else:
#             x= up
    
#         return x  


###TDA based on 1DConv#### 
# class TemporalAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, kernel_size=3, ff_multiplier=2, prompt_dim=128, adapter_layernorm_option="in", dropout=0.1):
#         """
#         dim: Dimension of the input tokens (from CLIP vision encoder).
#         reduction_factor: Reduction factor for the bottleneck in the adapter.
#         rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
#         prompt_dim: Dimensionality for the internal prompt.
#         adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
#         dropout: Dropout rate for regularization.
#         """
#         super(TemporalAdapter, self).__init__()

#         self.n_embd = dim
#         self.down_size = dim // reduction_factor
#         self.adapter_layernorm_option = adapter_layernorm_option

#         # LayerNorm setup (optional, based on adapter_layernorm_option)
#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(dim)

#         # Dynamic scaling mechanism
#         self.scale = nn.Linear(dim, 1)
       

#         # Down-projection to reduce dimensionality
#         self.down_proj = nn.Linear(dim, self.down_size)
#         self.activation = nn.GELU()
       
#        # 1D convolution for temporal modeling
#         self.conv1d = nn.Conv1d(
#             in_channels=self.down_size,
#             out_channels=self.down_size,
#             kernel_size=kernel_size,
#             padding=kernel_size // 2
#         )

      

#         # Up-projection to restore original dimensionality
#         self.up_proj = nn.Linear(self.down_size  , dim)

#         # Dropout for regularization
#         #self.dropout = nn.Dropout(dropout)

#         # Initialize weights for up-projection and dynamic scale (optional)
#         with torch.no_grad():
#             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.up_proj.weight)
#             nn.init.zeros_(self.down_proj.bias)
#             nn.init.zeros_(self.up_proj.bias)
#             nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.scale.bias)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

#     def forward(self, x, add_residual=True, residual=None):
#         """
#         Forward pass for dynamic temporal adapter with internal prompt generation.
#         x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
#         Returns:
#             - x: Adapted vision tokens with internal prompt concatenated.
#         """
#         # Save residual input if add_residual is True
#         residual = x 

#         # # Step 1: Apply LayerNorm if set to 'in'
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

      
#         # Step 2: Compute dynamic scaling factors for each token
#         dynamic_scale = F.relu(self.scale(x))  # (batch_size, seq_len, 1)
#         down = self.activation(self.down_proj(x))  # (B, T, D)
      

#         # Apply Conv1D over time (swap T and D axes)
#         down = down.transpose(1, 2)  # (B, D, T)
#         down = self.conv1d(down)
#         down = down.transpose(1, 2)  # (B, T, D)

#         up = self.up_proj(down)
#         up = up * dynamic_scale
      
   
#         if add_residual:
#             x = up + residual

#         else:
#             x= up
    
#         return x  









# class TemporalAdapter(nn.Module):
    # def __init__(self, dim, reduction_factor=8, adapter_layernorm_option="in"):
    #     """
    #     dim: Dimension of the input tokens (from CLIP vision encoder).
    #     reduction_factor: Reduction factor for the bottleneck in the adapter.
    #     rnn_type: Type of recurrent layer ('GRU', 'LSTM', or 'RNN').
    #     prompt_dim: Dimensionality for the internal prompt.#         adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
    #     dropout: Dropout rate for regularization.
    #     """
    #     super(TemporalAdapter, self).__init__()

    #     self.n_embd = dim
    #     self.down_size = dim // reduction_factor
    #     self.adapter_layernorm_option = adapter_layernorm_option

    #     # LayerNorm setup (optional, based on adapter_layernorm_option)
    #     self.adapter_layer_norm_before = None
    #     if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
    #         self.adapter_layer_norm_before = nn.LayerNorm(dim)

    #     # Dynamic scaling mechanism
    #     self.scale = nn.Linear(dim, 1)
       

    #     # Down-projection to reduce dimensionality
    #     self.down_proj = nn.Linear(dim, self.down_size)
    #     self.activation = nn.GELU()
       
       
#         self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
#                                                      input_dim=96,
#                                                      depth=1,
#                                                      heads=8,
#                                                      mlp_dim=1024,
#                                                      dim_head=64)

#         # Up-projection to restore original dimensionality
#         self.up_proj = nn.Linear(self.down_size  , dim)

#         # Dropout for regularization
#         #self.dropout = nn.Dropout(dropout)

#         # Initialize weights for up-projection and dynamic scale (optional)
#         with torch.no_grad():
#             nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.up_proj.weight)
#             nn.init.zeros_(self.down_proj.bias)
#             nn.init.zeros_(self.up_proj.bias)
#             nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.scale.bias)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
#             nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

#     def forward(self, x, add_residual=True, residual=None):
#         """
#         Forward pass for dynamic temporal adapter with internal prompt generation.
#         x: Input token embeddings from vision encoder (batch_size, seq_len, dim).
#         Returns:
#             - x: Adapted vision tokens with internal prompt concatenated.
#         """
#         # Save residual input if add_residual is True
#         residual = x 

#         # # Step 1: Apply LayerNorm if set to 'in'
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

      
#         # Step 2: Compute dynamic scaling factors for each token
#         dynamic_scale = F.relu(self.scale(x))  # (batch_size, seq_len, 1)
#         #dynamic_scale = torch.sigmoid(self.scale(x))
#         #weighted_x = x * dynamic_scale

#         # Step 3: Down-projection (reduce dimensionality)
#         down = self.down_proj(x)
        
#         # Step 4: Apply recurrent network (LSTM/GRU/RNN) to capture temporal dynamics
#         down = self.temporal_net(down)
       
#         down = self.activation(down)
#         #down = self.dropout(down)
#         #down = nn.functional.dropout(down, p=self.dropout, training=self.training)
#         # Step 5: Up-projection (restore original dimensionality)
#         up = self.up_proj(down)
#         #up = self.second_norm_layer(up)
        
#         #dynamic_scale = F.sigmoid(self.scale(up)) 
#         # Step 6: Apply dynamic scaling to the tokens
#         up = up * dynamic_scale
#         #up = up * gate
#         # if self.adapter_layernorm_option == 'out':
#         #     up = self.adapter_layer_norm_before(up)
#         if add_residual:
#             x = up + residual

#         else:
#             x= up
    
#         return x  

# class JointSpatialTemporalAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, num_regions=4, rnn_type='GRU', adapter_layernorm_option="in"):
#         """
#         Joint Spatial-Temporal Adapter
#         Combines region-aware spatial attention with temporal dynamics.
#         """
#         super(JointSpatialTemporalAdapter, self).__init__()

#         self.n_embd = dim
#         self.down_size = dim // reduction_factor
#         self.num_regions = num_regions
#         self.adapter_layernorm_option = adapter_layernorm_option

#         # LayerNorm setup
#         # self.adapter_layer_norm_before = None
#         # if adapter_layernorm_option in ["in", "out"]:
#         #     self.adapter_layer_norm_before = nn.LayerNorm(dim)

#         # Spatial Component: Region-Aware Attention
#         self.region_attention = nn.Linear(dim, num_regions)  # Regional attention scores
#         self.spatial_down_proj = nn.Linear(dim, dim // reduction_factor)
#         self.spatial_activation = nn.GELU()

#         # Temporal Component: Recurrent Network
#         self.temporal_down_proj = nn.Linear(dim, self.down_size)
#         if rnn_type == 'LSTM':
#             self.rnn = nn.LSTM(self.down_size, self.down_size, batch_first=True)
#         elif rnn_type == 'GRU':
#             self.rnn = nn.GRU(self.down_size, self.down_size, batch_first=True)
#         else:
#             raise ValueError("Unsupported RNN type. Choose 'GRU' or 'LSTM'.")

#         self.temporal_activation = nn.GELU()

#         # Fusion: Combine Spatial and Temporal Features
#         self.fusion_gate = nn.Linear(2 * (dim // reduction_factor), dim // reduction_factor)
#         self.fusion_activation = nn.GELU()

#         # Up-projection to restore original dimensionality
#         self.up_proj = nn.Linear(self.down_size, dim)

#         # Dynamic Scaling
#         self.scale = nn.Linear(dim, 1)  # Dynamic scaling factor

#         # Frame Attention for Noise Resilience
#         self.frame_attention = nn.Linear(dim, 1)  # Compute frame importance scores

#         with torch.no_grad():
#             nn.init.kaiming_uniform_(self.region_attention.weight, a=math.sqrt(5))
#             nn.init.kaiming_uniform_(self.spatial_down_proj.weight, a=math.sqrt(5))
#             nn.init.kaiming_uniform_(self.temporal_down_proj.weight, a=math.sqrt(5))
#             nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
#             nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
#             nn.init.zeros_(self.region_attention.bias)
#             nn.init.zeros_(self.spatial_down_proj.bias)
#             nn.init.zeros_(self.temporal_down_proj.bias)
#             nn.init.zeros_(self.up_proj.bias)
#             nn.init.zeros_(self.scale.bias)

#     def forward(self, x, add_residual=True):
#         """
#         x: Input token embeddings (batch_size, seq_len, dim).
#         Returns:
#             - Adapted features with spatial-temporal integration (batch_size, seq_len, dim).
#         """
#         residual = x

#         # Apply LayerNorm if set to 'in'
#         # if self.adapter_layernorm_option == '':
#         #     x = self.adapter_layer_norm_before(x)

#         # Step 1: Frame Attention for Noise Resilience
#         frame_scores = F.softmax(self.frame_attention(x), dim=1)  # (batch_size, seq_len, 1)
#         weighted_x = x * frame_scores  # Weight input frames based on importance

#         # Step 2: Spatial Component - Region-Aware Attention
#         region_scores = F.softmax(self.region_attention(weighted_x), dim=1)  # (batch_size, seq_len, num_regions)
#         region_features = torch.einsum('bsd,bsr->brd', weighted_x, region_scores)  # (batch_size, num_regions, dim)
#         spatial_features = self.spatial_down_proj(region_features)  # (batch_size, num_regions, down_size)
#         spatial_features = self.spatial_activation(spatial_features)

#         # Step 3: Temporal Component
#         temporal_input = self.temporal_down_proj(weighted_x)  # Down-project for temporal processing
#         temporal_features, _ = self.rnn(temporal_input)  # (batch_size, seq_len, down_size)
#         temporal_features = self.temporal_activation(temporal_features)

#         # Step 4: Fusion of Spatial and Temporal Features
#         # Align spatial features for fusion
#         spatial_features = spatial_features.mean(dim=1, keepdim=True).expand_as(temporal_features)
#         fused_features = torch.cat([spatial_features, temporal_features], dim=-1)  # (batch_size, seq_len, 2 * down_size)
#         fused_features = self.fusion_activation(self.fusion_gate(fused_features))  # (batch_size, seq_len, down_size)

#         # Step 5: Up-projection
#         up_features = self.up_proj(fused_features)  # (batch_size, seq_len, dim)

#         # Step 6: Apply Dynamic Scaling after Up-Projection
#         scale_factors = F.relu(self.scale(weighted_x))  # (batch_size, seq_len, 1)
#         scaled_features = up_features * scale_factors

#         # Step 7: Residual Connection
#         if add_residual:
#             return scaled_features + residual
#         return scaled_features

# class SpatialAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8):
#         """
#         Spatial Adapter (SA) with Spatial Attention and Bottleneck Design.
        
#         Args:
#             dim: Feature dimension.
#             reduction_factor: Factor for reducing dimensionality in the bottleneck.
#         """
#         super(SpatialAdapter, self).__init__()

#         self.dim = dim
#         self.reduction_factor = reduction_factor

#         # Spatial Attention Mechanism
#         self.spatial_attention = nn.Sequential(
#             nn.Linear(dim, dim // reduction_factor),  # Reduce feature dimension
#             nn.GELU(),
#             nn.Linear(dim // reduction_factor, 1),    # Output attention scores
#             nn.Sigmoid()                              # Normalize attention to [0, 1]
#         )

#         # Bottleneck Projection Layers
#         self.down_proj = nn.Linear(dim, dim // reduction_factor)  # Down-projection
#         self.activation = nn.GELU()
#         self.up_proj = nn.Linear(dim // reduction_factor, dim)    # Up-projection

#     def forward(self, x):
#         """
#         Forward pass for the Spatial Adapter.

#         Args:
#             x: Input features of shape (batch_size, seq_len, dim).
#         Returns:
#             Refined spatial features with residual connection.
#         """
#         residual = x  # Save input for residual connection

#         # Step 1: Compute Spatial Attention
#         attention_scores = self.spatial_attention(x)  # (batch_size, seq_len, 1)
#         x_spatial = x * attention_scores  # Apply attention to input features

#         # Step 2: Bottleneck Projection
#         x_down = self.down_proj(x_spatial)  # Reduce dimensionality
#         x_down = self.activation(x_down)    # Non-linearity
#         x_up = self.up_proj(x_down)         # Restore original dimensionality

#         # Step 3: Residual Connection
#         output = residual + x_up

#         return output


# class TemporalAdapter(nn.Module):
#     def __init__(self, dim, reduction_factor=8, rnn_type='GRU', adapter_layernorm_option="in"):
#         """
#         Improved Temporal Adapter (TDA++) with:
#         - Dual GRU for multi-scale temporal modeling (fine and coarse granularity).
#         - Temporal Importance Weighting (TIW) for key frame emphasis.
#         - Dynamic scaling for adaptive token refinement.

#         Args:
#             dim: Dimension of the input tokens (from CLIP vision encoder).
#             reduction_factor: Reduction factor for the bottleneck.
#             rnn_type: Type of RNN ('GRU', 'LSTM', 'RNN').
#             adapter_layernorm_option: Option for LayerNorm ('in', 'out', or None).
#         """
#         super(TemporalAdapter, self).__init__()

#         self.n_embd = dim
#         self.down_size = dim // reduction_factor
#         self.adapter_layernorm_option = adapter_layernorm_option

#         # LayerNorm before adapter if set to 'in'
#         self.adapter_layer_norm_before = nn.LayerNorm(dim) if adapter_layernorm_option == "in" else None

#         # Temporal Importance Weighting (TIW) - Learnable frame importance scores
#         self.frame_importance = nn.Linear(dim, 1)  

#         # Down-projection for dimensionality reduction
#         self.down_proj = nn.Linear(dim, self.down_size)
#         self.activation = nn.GELU()

#         # Dual GRU for multi-scale temporal dynamics
#         self.fine_gru = nn.GRU(self.down_size, self.down_size, batch_first=True)    # Fine-grained GRU
#         self.coarse_gru = nn.GRU(self.down_size, self.down_size // 2, batch_first=True)  # Coarse-grained GRU

#         # Up-projection to restore original dimensionality
#         self.up_proj = nn.Linear(self.down_size + self.down_size // 2, dim)

#         # Dynamic scaling for adaptive feature adjustment
#         self.dynamic_scale = nn.Linear(dim, 1)

#     def forward(self, x, add_residual=True):
#         """
#         Forward pass for TDA++.

#         Args:
#             x: Input token embeddings (batch_size, seq_len, dim).
#             add_residual: Whether to include the residual connection.
#         Returns:
#             x: Refined token embeddings.
#         """
#         residual = x

#         # Step 1: Optional LayerNorm
#         if self.adapter_layer_norm_before:
#             x = self.adapter_layer_norm_before(x)

#         # Step 2: Temporal Importance Weighting (TIW)
#         importance_scores = torch.softmax(self.frame_importance(x), dim=1)  # Frame-wise importance (normalized)
#         x_weighted = x * importance_scores  # Emphasize key frames

#         # Step 3: Down-projection
#         down = self.down_proj(x_weighted)
#         down = self.activation(down)

#         # Step 4: Multi-Scale GRU (Fine and Coarse Temporal Modeling)
#         fine_out, _ = self.fine_gru(down)           # Fine-grained temporal modeling
#         coarse_out, _ = self.coarse_gru(down)       # Coarse-grained temporal modeling

#         # Combine fine and coarse features
#         combined_temporal = torch.cat([fine_out, coarse_out], dim=-1)

#         # Step 5: Up-projection to restore dimensionality
#         up = self.up_proj(combined_temporal)

#         # Step 6: Dynamic Scaling
#         dynamic_scale = torch.sigmoid(self.dynamic_scale(x))  # Dynamic scaling factor
#         up = up * dynamic_scale

#         # Step 7: Residual Connection
#         if add_residual:
#             x = up + residual
#         else:
#             x = up

#         return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, adapter_type: str = 'text'):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # Choose the correct adapter based on the adapter_type
        if adapter_type == 'text':
            self.adapter_pre_attn = None#Adapter(d_model)  # Text Adapter before attention
            self.adapter_pre_mlp = Adapter(d_model)   # Text Adapter before MLP
        elif adapter_type == 'vision':
            self.adapter_pre_attn = None#TemporalAdapter(d_model)   # Temporal Adapter before attention for vision
            self.adapter_pre_mlp = None#TemporalAdapter(d_model)   # Temporal Adapter before MLP for vision
        else:
            raise ValueError("adapter_type must be 'text' or 'vision'")

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
    # Apply attention mechanism first, then apply adapter after
        x = x + self.attention(self.ln_1(x))
        x = self.adapter_pre_attn(x)
        # if self.adapter_pre_attn is not None:
        #  x = self.adapter_pre_attn(x)

        # Apply MLP first, then apply adapter after
        x = x + self.mlp(self.ln_2(x))
        if self.adapter_pre_mlp is not None:
         x = self.adapter_pre_mlp(x)
    
        return x

class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask
     
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.add_prompt:
            if not self.text_layer:
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
      
        x = x + self.mlp(self.ln_2(x))
        
        return x


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0, adapter_type: str = 'text', shared_adapter=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        shared_adapters = Adapter(d_model)
        

       # Choose the correct adapter based on the adapter_type
        if adapter_type == 'text':
            self.adapter_pre_attn = shared_adapters#Adapter(d_model)#shared_adapters#ContextualAttentionAdapter(d_model)#shared_adapters#Adapter(d_model)  # Text Adapter before attention
            self.adapter_pre_mlp = Adapter(d_model)#shared_adapter#Adapter(d_model)   # Text Adapter before MLP
            #self.adapter_pre_mhsa = None#Adapter(d_model)
        elif adapter_type == 'vision':
            self.adapter_pre_attn = shared_adapters#Adapter(d_model)#shared_adapters  #shared_adapters#None#RegionAwareAdapter(d_model)#SpatialAttentionAdapter(d_model)#shared_adapters#SA_Adapter(d_model) #Adapter(d_model)#None#SAAdapter(d_model)#shared_adapters#Spatio_Adapter(d_model)#shared_adapters#TemporalAdapter(d_model)   # Temporal Adapter before attention for vision
            self.adapter_pre_mlp =  TemporalAdapter(d_model)#Adapter(d_model) # Temporal Adapter before MLP for vision
            #self.adapter_pre_mhsa = RegionAwareAdapter(d_model)
            #self.adapter_pre_mlp = TemporalAdapter(d_model)   # Temporal Adapter before MLP for vision
        else:
            raise ValueError("adapter_type must be 'text' or 'vision'")
        
        #self.shared_adapter = shared_adapter

        self.compound_prompt_nctx = design_details['maple_length']
        self.first_layer = (i == 0)
        self.non_linear_func = nn.GELU()

       
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]

    
    def forward(self, inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                if not self.text_layer:  # Vision layer
                    if not (counter > len(compound_prompts_deeper) - 1):
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        visual_context = compound_prompts_deeper[counter]
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        x = torch.cat([prefix, visual_context], dim=0)
                        counter += 1
                else:  # Text layer
                    if not (counter > len(compound_prompts_deeper) - 1):
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        counter += 1
    
    # Apply attention mechanism first, then apply adapter after
        # x = x + self.attention(self.ln_1(x))
        # if self.adapter_pre_attn is not None:
        #  x = self.adapter_pre_attn(x)
      
        # # Apply MLP first, then apply adapter after
        # x = x + self.mlp(self.ln_2(x))
        # if self.adapter_pre_mlp is not None:
        #  x = self.adapter_pre_mlp(x)
    
        residual = x

        # if self.adapter_pre_attn is not None:
        #     x = self.adapter_pre_attn(x)
        # x = x + self.attention(self.ln_1(x))
    
        # if self.adapter_pre_mlp is not None:
        #    x = self.adapter_pre_mlp(x)
          
        # x = x + self.mlp(self.ln_2(x))   
       
        #### best way#######
        x_attn = self.attention(self.ln_1(x))  # Apply attention
        if self.adapter_pre_attn is not None:
            x_attn = self.adapter_pre_attn(x_attn)  # Apply adapter after attention
            
        x = residual + x_attn  # Apply residual connection

        # Apply MLP first, then the adapter, and finally apply the residual connection
        x_residual = x 
        x_mlp = self.mlp(self.ln_2(x))  # Apply MLP
        if self.adapter_pre_mlp is not None:
            x_mlp = self.adapter_pre_mlp(x_mlp)  # Apply adapter after MLP
        x = x_residual +  x_mlp  # Apply residual connection
        return [x, compound_prompts_deeper, counter]
        ##############################
        # if self.adapter_pre_mhsa is not None:
        #     x = self.adapter_pre_mhsa(x)

        # x_attn = self.attention(self.ln_1(x))  # Apply attention
        # if self.adapter_pre_attn is not None:
        #     x_attn = self.adapter_pre_attn(x_attn)
        #   # Apply adapter after attention
        # x = residual + x_attn  # Apply residual connection
        # # Apply MLP first, then the adapter, and finally apply the residual connection
        # x_residual = x 
        # x_mlp = self.mlp(self.ln_2(x))  # Apply MLP
        # if self.adapter_pre_mlp is not None:
        #     x_shared = self.adapter_pre_mlp(x_mlp)  # Apply adapter after ML
        # x = x_residual +  x_shared  # Apply residual connection
        # return [x, compound_prompts_deeper, counter]

        



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None,  adapter_type: str = 'text', shared_adapter=None): #, shared_adapter=None
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer=text_layer, i=i, adapter_type=adapter_type, shared_adapter=shared_adapter)
                  for i in range(layers)]) #, shared_adapter=shared_adapter[i]
        else:
            # Corresponds to default CoOp or CoCoOp
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details, shared_adapter=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details, adapter_type='vision', shared_adapter=shared_adapter)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details, shared_adapter=None):#, shared_adapter=None
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details, adapter_type='vision', shared_adapter=shared_adapter)#, shared_adapter=shared_adapter

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']

        #self.shared_adapters = Adapter(embed_dim)
        #     for i in range(transformer_layers)])
        
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer == "MaPLe":
                self.visual = VisionTransformer_MaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                    #shared_adapter = self.shared_adapters

                )
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details,
            adapter_type='text' 
            #shared_adapter = self.shared_adapters
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict, strict=False)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
