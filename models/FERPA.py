import torch
from torch import nn
from models.MLPrompt import MultiModalPromptLearner, TextEncoder


# class TextRefinementLayer(nn.Module):
#     def __init__(self, dim, reduction_factor=8, attention=False):
#         super().__init__()
#         self.reduction_factor = reduction_factor
#         self.attention = attention

#         # Bottleneck projection
#         self.down_proj = nn.Linear(dim, dim // reduction_factor)
#         self.activation = nn.GELU()
#         self.up_proj = nn.Linear(dim // reduction_factor, dim)

#         # Optional attention mechanism
#         if self.attention:
#             self.attention_layer = nn.Linear(dim, 1)

#     def forward(self, x):
#         residual = x

#         # Optional attention for text refinement
#         if self.attention:
#             attention_scores = torch.softmax(self.attention_layer(x), dim=1)
#             x = x * attention_scores

#         # Bottleneck refinement
#         x = self.down_proj(x)
#         x = self.activation(x)
#         x = self.up_proj(x)

#         return x + residual

# class DynamicTextAdapter(nn.Module):
#     def __init__(self, text_dim, vision_dim, reduction_factor=8):
#         """
#         Dynamic Text Adapter for refining text embeddings with visual context.

#         Args:
#             text_dim: Dimensionality of the text features.
#             vision_dim: Dimensionality of the vision features.
#             reduction_factor: Bottleneck reduction factor.
#         """
#         super(DynamicTextAdapter, self).__init__()
#         self.text_down_proj = nn.Linear(text_dim, text_dim // reduction_factor)
#         self.vision_proj = nn.Linear(vision_dim, text_dim // reduction_factor)
#         self.activation = nn.GELU()
#         self.text_up_proj = nn.Linear(text_dim // reduction_factor, text_dim)
        
#         # Scaling mechanism
#         self.scale = nn.Linear(text_dim, 1)

#     def forward(self, text_features, vision_features):
#         """
#         Refine text features with vision features.

#         Args:
#             text_features: Text embeddings, shape (batch_size_t, text_dim).
#             vision_features: Vision embeddings, shape (batch_size_v, vision_dim).

#         Returns:
#             Refined text embeddings.
#         """
#         # Align dimensions: Expand or average vision features to match text features
#         if vision_features.size(0) < text_features.size(0):
#             vision_features = vision_features.unsqueeze(1).expand(-1, text_features.size(0), -1)
#             vision_features = vision_features.mean(dim=0)  # Align visual features
#         elif text_features.size(0) < vision_features.size(0):
#             text_features = text_features.mean(dim=0, keepdim=True).expand(vision_features.size(0), -1)

#         # Step 1: Project to lower dimension
#         text_proj = self.text_down_proj(text_features)
#         vision_proj = self.vision_proj(vision_features)

#         # Step 2: Combine text and vision features
#         combined_features = text_proj + vision_proj
#         combined_features = self.activation(combined_features)

#         # Step 3: Scaling mechanism
#         scaling_factor = torch.sigmoid(self.scale(text_features))
#         refined_text = self.text_up_proj(combined_features) * scaling_factor

#         # Step 4: Residual connection
#         output = refined_text + text_features
#         return output



# class GenerateModel(nn.Module):
#     def __init__(self, input_text, clip_model, args):
#         super().__init__()
#         self.args = args
#         self.input_text = input_text

#         # Initialize the MultiModalPromptLearner
#         self.prompt_learner = MultiModalPromptLearner(input_text, clip_model, args)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         self.text_refinement = Adapter(dim=512, reduction_factor=8)

#         # Use the TextEncoder from MaPLe
#         self.text_encoder = TextEncoder(clip_model)
#         self.dtype = clip_model.dtype

#         # CLIP's visual encoder
#         self.image_encoder = clip_model.visual
#         self.clip_model = clip_model

#         # Add a learnable alignment layer for image and text features
#         # self.image_alignment_layer = nn.Linear(clip_model.visual.output_dim, 512).cuda()
#         # self.text_alignment_layer = nn.Linear(clip_model.transformer.width, 512).cuda()

#         # Cross-modal attention layer (optional for enhanced alignment)
#         #self.cross_modal_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

#         # Whitening layer for feature preprocessing
#         #self.feature_whitening = nn.BatchNorm1d(512, affine=False).cuda()

#         # nn.init.xavier_uniform_(self.image_alignment_layer.weight)
#         # nn.init.zeros_(self.image_alignment_layer.bias)
#         # nn.init.xavier_uniform_(self.text_alignment_layer.weight)
#         # nn.init.zeros_(self.text_alignment_layer.bias)

#         # Cross-Modal Alignment Layer
        

#         self.to(self.dtype)

#     def forward(self, image):
#         # Process video frames
#         n, t, c, h, w = image.shape
#         image = image.contiguous().view(-1, c, h, w).type(self.dtype).cuda()

#         # Generate prompts
#         prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

#         # Process image frames with the image encoder
#         image_features = self.image_encoder(image, shared_ctx, deep_compound_prompts_vision).cuda()
#         image_features = image_features.contiguous().view(n, t, -1)  # [batch_size, time_steps, feature_dim]
#         image_features = torch.mean(image_features, dim=1)  # Temporal mean pooling

#         # Normalize image features
       
#         #image_features = self.image_alignment_layer(image_features)  # Alignment layer
#         #image_features = self.feature_whitening(image_features)  # Whitening step
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#         # Encode text features
#         text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text).cuda()
#         text_features = self.text_refinement(text_features)
  
#         #text_features = self.text_alignment_layer(text_features)  # Alignment layer
#         #text_features = self.feature_whitening(text_features)  # Whitening step
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         # Optional: Apply cross-modal attention
#         # image_features, _ = self.cross_modal_attention(image_features.unsqueeze(1), 
#         #                                                 text_features.unsqueeze(1), 
#         #                                                 text_features.unsqueeze(1))
#         # image_features = image_features.squeeze(1)

#         # Compute cosine similarity
#          # Step 5: Cross-Modal Alignment
    

#         output = image_features @ text_features.t() / 0.01

#         return output


class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text

        # Initialize the MultiModalPromptLearner
        self.prompt_learner = MultiModalPromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # Use the TextEncoder from MaPLe
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

        # CLIP's visual encoder
        self.image_encoder = clip_model.visual
        self.clip_model = clip_model

        self.to(self.dtype)

    def forward(self, image):
        # Process video frames
        #n, t, c, h, w = image.shape
        
        if len(image.shape) == 5:  # 5D input: [n, t, c, h, w]
            n, t, c, h, w = image.shape
        elif len(image.shape) == 4:  # 4D input: [n, c, h, w]
            n, c, h, w = image.shape
            t = 1
            image = image.unsqueeze(1)  # Add temporal dimension
        else:
            raise ValueError(f"Unexpected input shape: {image.shape}")
        image = image.contiguous().view(-1, c, h, w).type(self.dtype).cuda()

        # Generate prompts
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

        # Process image frames with the image encoder
        image_features = self.image_encoder(image, shared_ctx, deep_compound_prompts_vision).cuda()
        image_features = image_features.contiguous().view(n, t, -1)  # [batch_size, time_steps, feature_dim]
        image_features = torch.mean(image_features, dim=1)  # Temporal mean pooling
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode text features
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text).cuda()
        
        # Normalize text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        output = image_features @ text_features.t() / 0.01

        return output




# class GenerateModel(nn.Module):
#     def __init__(self, input_text, clip_model, args):
#         super().__init__()
#         self.args = args
#         self.input_text = input_text

#         # Initialize the MultiModalPromptLearner
#         self.prompt_learner = MultiModalPromptLearner(input_text, clip_model, args)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts

#         # Use the TextEncoder from MaPLe
#         self.text_encoder = TextEncoder(clip_model)
#         self.dtype = clip_model.dtype

#         # CLIP's visual encoder
#         self.image_encoder = clip_model.visual
#         self.clip_model = clip_model

#         # self.projection_dim = 512  # Set projection dimension (e.g., 512 or 256)
#         # self.image_projection = nn.Linear(clip_model.visual.output_dim, self.projection_dim)
#         # text_output_dim = clip_model.transformer.width if hasattr(clip_model, 'transformer') else clip_model.text_encoder.output_dim
#         # self.text_projection = nn.Linear(text_output_dim, self.projection_dim)
      

#         # Temporal Attention Pooling Module
#         #self.temporal_attention_pooling = TemporalAttentionPooling(feature_dim=512)  # 512 is the feature size

#         self.to(self.dtype)

#     def forward(self, image):
#         # Process video frames
#         n, t, c, h, w = image.shape
#         # Flatten the temporal dimension
#         image = image.contiguous().view(-1, c, h, w).type(self.dtype).cuda()

#         # Generate prompts
#         prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

#         # Process each frame with the image encoder
#         image_features = self.image_encoder(image, shared_ctx, deep_compound_prompts_vision).cuda()
#         # Reshape to [batch_size, time_steps, feature_dim]
#         image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
#         image_features = torch.mean(image_features, dim=1)
#         #image_features = self.image_projection(image_features)
        
#         # Apply Temporal Attention Pooling
#         #image_features = self.temporal_attention_pooling(image_features)  # [n, 512]

#         # Normalize the pooled image features
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         #projection_layer = nn.Linear(536, 512).cuda()  # Project image features to 512 dimensions
#         #image_features = projection_layer(image_features).cuda()

#         # Encode text features
#         text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text).cuda()
#         #text_features = self.text_projection(text_features)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True).cuda()
        
#         # Compute similarity
#         output = image_features @ text_features.t() / 0.01

#         return output

# class GenerateModel(nn.Module):
#     def __init__(self, input_text, clip_model, args):
#         super().__init__()
#         self.args = args
#         self.input_text = input_text

#         # Initialize the MultiModalPromptLearner
#         self.prompt_learner = MultiModalPromptLearner(input_text, clip_model, args)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts

#         # Use the TextEncoder from MaPLe
#         self.text_encoder = TextEncoder(clip_model)
#         self.dtype = clip_model.dtype

#         # CLIP's visual encoder
#         self.image_encoder = clip_model.visual
#         self.clip_model = clip_model

#         # Define the Emotion-Conditional Text Adapter (ECTA)
#         self.emotion_conditional_text_adapter = EmotionConditionalTextAdapter(text_dim=512, vision_dim=512)  # Assuming text_dim and vision_dim to be 512

#         # To make sure model parameters are using the same dtype
#         self.to(self.dtype)

#     def forward(self, image):
#         # Process video frames
#         n, t, c, h, w = image.shape
#         # Flatten the temporal dimension
#         image = image.contiguous().view(-1, c, h, w).type(self.dtype).cuda()

#         # Generate prompts
#         prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

#         # Process each frame with the image encoder
#         image_features = self.image_encoder(image, shared_ctx, deep_compound_prompts_vision).cuda()
#         # Reshape to [batch_size, time_steps, feature_dim]
#         image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]

#         # Mean pooling across the temporal dimension to get aggregated image features
#         image_features = torch.mean(image_features, dim=1)  # Shape: [n, 512]

#         # Normalize the pooled image features
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#         # Extract the emotion features from the image (this will condition the text)
#         emotion_features = image_features  # You can refine this further if you have specific emotion features

#         # Encode text features
#         text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text).cuda()

#         # Apply Emotion-Conditional Text Adapter (ECTA) to modulate text embeddings based on visual emotion features
#         text_features = self.emotion_conditional_text_adapter(text_features, emotion_features).cuda()

#         # Normalize the text features
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True).cuda()

#         # Compute similarity
#         output = image_features @ text_features.t() / 0.01

#         return output


# class EmotionConditionalTextAdapter(nn.Module):
#     def __init__(self, text_dim, vision_dim):
#         super(EmotionConditionalTextAdapter, self).__init__()
#         # Linear projection from visual (emotion) features to text space
#         self.vision_to_text_adapter = nn.Linear(vision_dim, text_dim)
#         # Linear projection within the text space
#         self.text_proj = nn.Linear(text_dim, text_dim)

#     def forward(self, text_x, emotion_features):
#         """
#         Modulate the text embeddings based on emotion features from vision encoder.
#         Args:
#             text_x: Text embeddings [batch_size, text_dim]
#             emotion_features: Emotion-related features from the vision encoder [batch_size, vision_dim]
#         """
#         # Project the emotion features from vision space into text space
#         emotion_in_text_space = self.vision_to_text_adapter(emotion_features)

#         # Modulate the text embeddings by adding the projected emotion features
#         text_x = text_x + self.text_proj(emotion_in_text_space)

#         return text_x


# import torch
# from torch import nn
# from models.MLPrompt import MultiModalPromptLearner, TextEncoder

# class GenerateModel(nn.Module):
#     def __init__(self, input_text, clip_model, args):
#         super().__init__()
#         self.args = args
#         self.input_text = input_text

#         # Initialize the MultiModalPromptLearner
#         self.prompt_learner = MultiModalPromptLearner(input_text, clip_model, args)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts

#         # Use the TextEncoder from MaPLe
#         self.text_encoder = TextEncoder(clip_model)
#         self.dtype = clip_model.dtype

#         # CLIP's visual encoder
#         self.image_encoder = clip_model.visual
#         self.clip_model = clip_model

#         self.to(self.dtype)

#     def forward(self, image):
#         # Process video frames
#         n, t, c, h, w = image.shape
#         #print(f"Image shape: {image.shape}")
#         image = image.contiguous().view(-1, c, h, w).type(self.dtype)

#         # Generate prompts
#         prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

#         # Process each frame with the image encoder
#         image_features = self.image_encoder(image, shared_ctx, deep_compound_prompts_vision)
#         #print(f"Output shape from CLIP visual encoder: {image_features.shape}")
#         image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
#         #print(f"Reshape Output shape from CLIP visual encoder: {image_features.shape}")

#         image_features = torch.mean(image_features, dim=1)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#         # Encode text features
#         text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
       
#         # Compute similarity
#         output = image_features @ text_features.t() / 0.01

#         return output











# import torch
# from torch import nn
# from models.MLPrompt import MultiModalPromptLearner, TextEncoder

# class GenerateModel(nn.Module):
#     def __init__(self, input_text, clip_model, args):
#         super().__init__()
#         self.args = args
#         self.input_text = input_text

#         # Initialize the MultiModalPromptLearner
#         self.prompt_learner = MultiModalPromptLearner(input_text, clip_model, args)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts

#         # Use the TextEncoder from MaPLe
#         self.text_encoder = TextEncoder(clip_model)
#         self.dtype = clip_model.dtype

#         # CLIP's visual encoder
#         self.image_encoder = clip_model.visual
#         self.clip_model = clip_model

#         self.to(self.dtype)

#     def forward(self, image):
#         # Process video frames
#         n, t, c, h, w = image.shape
#         #print(f"Image shape: {image.shape}")
#         image = image.contiguous().view(-1, c, h, w).type(self.dtype)

#         # Generate prompts
#         prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

#         # Process each frame with the image encoder
#         image_features = self.image_encoder(image, shared_ctx, deep_compound_prompts_vision)
#         #print(f"Output shape from CLIP visual encoder: {image_features.shape}")
#         image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
#         #print(f"Reshape Output shape from CLIP visual encoder: {image_features.shape}")

#         image_features = torch.mean(image_features, dim=1)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#         # Encode text features
#         text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
       
#         # Compute similarity
#         output = image_features @ text_features.t() / 0.01

#         return output
    
    # def get_adapter_params(self):
    #     adapter_params = []
    #     for name, param in self.clip_model.named_parameters():
    #         if 'adapter' in name:
    #             param.requires_grad = True  # Ensure requires_grad is set to True
    #             adapter_params.append(param)
    #             print(f"Adapter parameter found: {name}, requires_grad set to True")
    #         else:
    #             param.requires_grad = False  # Optionally, you can ensure others are False
    #     return adapter_params

    
    
# class Adapter(nn.Module):
#     def __init__(self, input_dim, reduction_factor=4):
#         super(Adapter, self).__init__()
#         reduced_dim = input_dim // reduction_factor
#         self.down_proj = nn.Linear(input_dim, reduced_dim)
#         self.activation = nn.GELU()  # Replace ReLU with GELU
#         self.up_proj = nn.Linear(reduced_dim, input_dim)
#         self.residual = nn.Identity()  # Identity for residual connection

#     def forward(self, x):
#         residual = self.residual(x)
#         x = self.down_proj(x)
#         x = self.activation(x)
#         x = self.up_proj(x)
#         return x + residual