import torch
from torch import nn
from models.MLPrompt import MultiModalPromptLearner, TextEncoder

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
