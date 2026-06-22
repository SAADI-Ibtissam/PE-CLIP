


import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from dataloader.video_transform import *
import torch.nn as nn
import torch.nn.parallel
import math
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import torch.utils.data
import thop
from thop import profile
import torch.utils.data.distributed
from models.FERPA import GenerateModel
from fmix import  sample_mask
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
import matplotlib
matplotlib.use('Agg')
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
from models.clip1 import clip
import warnings
import torchvision.transforms
from FERPAmainmxp import RecorderMeter
warnings.filterwarnings("ignore", category=UserWarning)
from models.Text import *


from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=1)


parser.add_argument('--lr-adapter', type=float, default=2e-4)
parser.add_argument('--lr-prompt-learner', type=float, default=3e-5)

parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--warmup_epochs', type=int, default=2, help='number of warmup epochs.')
parser.add_argument('--model-name', type=str, default="ViT-B/16")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int)
parser.add_argument('--context-number', type=int, default=2)
parser.add_argument('--num-cont', type=int, default=2)
parser.add_argument('--init-text', type=str, default='False')
parser.add_argument('--resume', type=str, default='False')
parser.add_argument('--prompt-depth', type=int, default=9)
parser.add_argument('--class-token-position', type=str, default="end")
parser.add_argument('--class-specific-contexts', type=str, default='True')
parser.add_argument('--load_and_tune_prompt_learner', type=str ,default='True')
parser.add_argument('--text-type', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--exper-name', type=str)
parser.add_argument('--temporal-layers', type=int, default=1)

parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
parser.add_argument('--image_path', type=str, default='./test/00094_00022.jpg',
                        help='Input image path')
parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
args = parser.parse_args()


# Define the RecorderMeter dummy class
class RecorderMeter:
    def __init__(self, *args, **kwargs):
        pass

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap   + np.float32(img) 
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



# def show_mask_on_image(img, mask):
#     """
#     Overlay a heatmap on the input image to visualize attention.
    
#     Args:
#         img (numpy.ndarray): The input image (H x W x C) in BGR format (OpenCV format).
#         mask (numpy.ndarray): The attention mask (H x W), normalized to [0, 1].
        
#     Returns:
#         numpy.ndarray: The image with the heatmap overlay.
#     """
#     # Ensure input image is in the range [0, 1]
#     img = np.float32(img) / 255.0
    
#     # Normalize the mask to [0, 1]
#     mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
#     # Apply the colormap
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255.0  # Convert to [0, 1]
    
#     # Blend the heatmap and the image
#     # Adjust these blending ratios to emphasize the heatmap or the image
#     cam = heatmap * 0.6 + img * 0.4
#     cam = cam / cam.max()  # Normalize the result
    
#     return np.uint8(255 * cam)




if args.dataset == "FERV39K" or args.dataset == "DFEW":
    number_class = 7
    class_names = class_names_7
    class_names_with_context = class_names_with_context_7
    class_descriptor = class_descriptor_7

device = "cuda:0" 
if __name__ == '__main__':

    os.makedirs('./Attentionmap', exist_ok=True)

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,}

    log_txt_path = './log/' + 'DFEW-' +  '-log.txt'
    def load_clip_to_cpu(args):
        backbone_name = args.model_name
        
        # Set the local path where you've saved the downloaded model
        local_model_path = "F:\KmuProj2\Code4\models\ViT-B-16.pt"  # Update this path accordingly

        try:
            # Try to load the model as a TorchScript JIT archive
            model = torch.jit.load(local_model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            # If the above fails, load it as a state dictionary
            state_dict = torch.load(local_model_path, map_location="cpu")

        design_details = {
            "trainer": 'MaPLe',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": args.num_cont
        }
        
        # Build the model using the loaded state_dict or the JIT model's state_dict
        model = clip.build_model(state_dict or model.state_dict(), design_details)

        return model
    
    CLIP_model = load_clip_to_cpu(args)
    CLIP_model.float()

    # Prepare input text for the model
    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor

    # Generate the model and load the pre-trained weights
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    model = torch.nn.DataParallel(model).to(device)



    
    #print(model)
    # for name, layer in model.module.clip_model.visual.named_modules():
    #  layer.register_forward_hook(hook_fn)

  
    #shufflenetww = torch.load('shufflnetw1_Ourmodel/1/Test_model.t7')
    #efficientvitww = torch.load('efficientvitw2_Ourmodel/1/Test_model.t7')
    ourmodel = torch.load('./log/fold1/Fullmodelbest/DFEW-2501041852GRUoursmodel-set1-model_best.pth')#Ours
    #ourmodel = torch.load('./log/fold1/baseline/withaumaple/DFEW-2501051345firstconfgofAdap-set1-model_best.pth')#Baseline
    #ourmodel = torch.load('./log/fold1/withoutTDA(sharedadapters/DFEW-2412280443withoutTDA-set1-model_best.pth')#Baseline+Sha
  
    model.load_state_dict(ourmodel['state_dict'],strict=True) 


    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Assuming 16-frame video input

    #GFlOPs calculation
   
    # def count_trainable_flops(model, dummy_input):
    #     model.eval()
        
    #     # Ensure only trainable parts are profiled
    #     def forward_hook(module, input, output):
    #         if not any(p.requires_grad for p in module.parameters()):
    #             return  # Skip frozen layers
    #         return thop.count_ops(module, input, output)

    #     flops, _ = profile(model.module if isinstance(model, torch.nn.DataParallel) else model,
    #                     inputs=(dummy_input,), custom_ops={torch.nn.Module: forward_hook})
        
    #     gflops = flops / 1e9  # Convert FLOPs to GFLOPs
    #     return gflops
    
    # gflops = count_trainable_flops(model, dummy_input)
    # print(f"GFLOPs (Trainable Parts Only): {gflops:.2f}") 
    import time

    with torch.no_grad():
        start_time = time.time()
        _ = model(dummy_input)
        end_time = time.time()

    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference Time per Video Sequence: {inference_time:.2f} ms")

    
    transform = torchvision.transforms.Compose([
         #torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
       
    ])
    
#     transform = torchvision.transforms.Compose([
#     #torchvision.transforms.ToPILImage(),
#     torchvision.transforms.Resize((224, 224)),  # Resize to 224x224
#     torchvision.transforms.Lambda(lambda x: x[:, :, ::-1] if isinstance(x, np.ndarray) else x),  # Reverse channels if needed
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
# ])

    torch.backends.cudnn.enabled = False
    # Directory containing test images
    input_folder = './test'  # Change to your folder path
    output_folder = './output_maps'  # Folder to save output images
    os.makedirs(output_folder, exist_ok=True)

    # Ensure device availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Iterate through all image files in the input folder
    for image_path in glob.glob(os.path.join(input_folder, '*.jpg')):
        print(f"Processing image: {image_path}")

        # Load the image
        rgb_img = Image.open(image_path).convert('RGB')

        # Apply transformations
        input_tensor = transform(rgb_img)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        print('Input tensor shape:', input_tensor.shape)

        # Perform Attention Rollout
        if args.category_index is None:
            print("Doing Attention Rollout")
            #attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
            attention_rollout = VITAttentionRollout(model, head_fusion="min", discard_ratio=0.9)
            mask = attention_rollout(input_tensor)
            output_name = f"attention_rolloutours_{os.path.basename(image_path)}"
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            output_name = f"grad_rollout_{os.path.basename(image_path)}"

        # Process and save attention map
        np_img = np.array(rgb_img)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)

        # Save the input image and attention map
        #cv2.imwrite(os.path.join(output_folder, f"input_{os.path.basename(image_path)}"), np_img)
        cv2.imwrite(os.path.join(output_folder, output_name), mask)

    print(f"Processing completed. Output maps saved in {output_folder}.")
    # ## select part of test data to gen
    # #for p in glob.glob('./test/*.jpg'):
     
    # rgb_img = Image.open(args.image_path)
    # #rgb_img = cv2.imread(p, 1)[:, :, ::-1].copy()
    # input_tensor = transform(rgb_img)
    # input_tensor = input_tensor.unsqueeze(0)
    #         #input_tensor = input_tensor.unsqueeze(1)
    # input_tensor = input_tensor.to(device)
    # print('okkkkkkkkkk',input_tensor.shape)
            
    # if args.category_index is None:
    #             print("Doing Attention Rollout")
    #             attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
    #                 discard_ratio=args.discard_ratio)
    #             mask = attention_rollout(input_tensor)
    #             name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    # else:
    #             print("Doing Gradient Attention Rollout")
    #             grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
    #             mask = grad_rollout(input_tensor, args.category_index)
    #             name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
    #                 args.discard_ratio, args.head_fusion)
                
    # np_img = np.array(rgb_img)[:, :, ::-1]
    # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    # mask = show_mask_on_image(np_img, mask)
    # cv2.imwrite("input.png", np_img)
    # print("Attention map saved as attention_map.png")
    # cv2.imwrite(name, mask)
    # cv2.waitKey(-1)
 

          


#For visualize conv layer (work well!)

# import os
# import glob
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import models
# from dataloader.video_transform import *
# import torch.nn as nn
# import torch.nn.parallel
# import math
# import torch.backends.cudnn as cudnn
# import torch.optim
# import argparse
# import torch.utils.data
# from thop import profile
# import torch.utils.data.distributed
# from models.FERPA import GenerateModel
# from fmix import  sample_mask
# import matplotlib
# matplotlib.use('Agg')
# from timm.scheduler import CosineLRScheduler
# import matplotlib.pyplot as plt
# import numpy as np
# import itertools
# import datetime
# from dataloader.video_dataloader import train_data_loader, test_data_loader
# from sklearn.metrics import confusion_matrix
# import tqdm
# from models.clip1 import clip
# import warnings
# import torchvision.transforms
# from FERPAmainmxp import RecorderMeter
# warnings.filterwarnings("ignore", category=UserWarning)
# from models.Text import *


# from pytorch_grad_cam import GradCAM, \
#                              ScoreCAM, \
#                              GradCAMPlusPlus, \
#                              AblationCAM, \
#                              XGradCAM, \
#                              EigenCAM, \
#                              EigenGradCAM

# from pytorch_grad_cam.utils.image import show_cam_on_image, \
#                                          deprocess_image, \
#                                          preprocess_image


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str)
# parser.add_argument('--workers', type=int, default=0)
# parser.add_argument('--epochs', type=int, default=30)
# parser.add_argument('--batch-size', type=int, default=8)


# parser.add_argument('--lr-adapter', type=float, default=2e-4)
# parser.add_argument('--lr-prompt-learner', type=float, default=3e-5)

# parser.add_argument('--weight-decay', type=float, default=0.01)
# parser.add_argument('--warmup_epochs', type=int, default=2, help='number of warmup epochs.')
# parser.add_argument('--model-name', type=str, default="ViT-B/16")
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--print-freq', type=int, default=10)
# parser.add_argument('--milestones', nargs='+', type=int)
# parser.add_argument('--context-number', type=int, default=2)
# parser.add_argument('--num-cont', type=int, default=2)
# parser.add_argument('--init-text', type=str, default='False')
# parser.add_argument('--resume', type=str, default='False')
# parser.add_argument('--prompt-depth', type=int, default=9)
# parser.add_argument('--class-token-position', type=str, default="end")
# parser.add_argument('--class-specific-contexts', type=str, default='True')
# parser.add_argument('--load_and_tune_prompt_learner', type=str ,default='True')
# parser.add_argument('--text-type', type=str)
# parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--exper-name', type=str)
# parser.add_argument('--temporal-layers', type=int, default=1)
# args = parser.parse_args()


# # Define the RecorderMeter dummy class
# class RecorderMeter:
#     def __init__(self, *args, **kwargs):
#         pass


# # def reshape_transform(tensor, height=14, width=14):
# #     result = tensor[:, 1 :  , :].reshape(tensor.size(0),
# #         height, width, tensor.size(2))

# #     # Bring the channels to the first dimension,
# #     # like in CNNs.
# #     result = result.transpose(2, 3).transpose(1, 2)
# #     return result

# def reshape_transform(tensor):
#     print(f"Activation shape inside reshape_transform: {tensor.shape}")

#     # Ensure tensor has the correct dimensions
#     if tensor.dim() == 2:  # [199, 768], Add batch dimension
#         tensor = tensor.unsqueeze(0)
#     elif tensor.dim() == 3 and tensor.shape[1] == 1:  # No spatial tokens
#         raise ValueError(f"No spatial tokens found in tensor: {tensor.shape}")

#     # Extract spatial tokens (ignore the class token at index 0)
#     spatial_tokens = tensor[:, 1:, :]  # [batch_size, num_spatial_tokens, features]
#     print(f"Spatial tokens shape: {spatial_tokens.shape}")

#     # Ensure valid number of spatial tokens
#     num_spatial_tokens = spatial_tokens.size(1)
#     if num_spatial_tokens <= 0:
#         raise ValueError(f"Invalid number of spatial tokens: {num_spatial_tokens}. Check model output.")

#     # Infer grid dimensions
#     height = int(num_spatial_tokens ** 0.5)
#     width = num_spatial_tokens // height
#     if height * width != num_spatial_tokens:
#         raise ValueError(
#             f"Cannot reshape {num_spatial_tokens} tokens into a grid. "
#             f"Closest dimensions: {height} x {width}."
#         )

#     print(f"Reshaping into grid: {height} x {width}")
#     return spatial_tokens.reshape(tensor.size(0), height, width, tensor.size(2)).permute(0, 3, 1, 2)





# def hook_fn(module, input, output):
#     if hasattr(output, "shape"):
#         print(f"Output shape of {module}: {output.shape}")
#     else:
#         print(f"Output is not a tensor, type: {type(output)}")

# if args.dataset == "FERV39K" or args.dataset == "DFEW":
#     number_class = 7
#     class_names = class_names_7
#     class_names_with_context = class_names_with_context_7
#     class_descriptor = class_descriptor_7

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# if __name__ == '__main__':

#     os.makedirs('./Attentionmap', exist_ok=True)

#     methods = \
#         {"gradcam": GradCAM,
#          "scorecam": ScoreCAM,
#          "gradcam++": GradCAMPlusPlus,
#          "ablationcam": AblationCAM,
#          "xgradcam": XGradCAM,
#          "eigencam": EigenCAM,
#          "eigengradcam": EigenGradCAM,}

  
#     def load_clip_to_cpu(args):
#         backbone_name = args.model_name
        
#         # Set the local path where you've saved the downloaded model
#         local_model_path = "F:\KmuProj2\Code4\models\ViT-B-16.pt"  # Update this path accordingly

#         try:
#             # Try to load the model as a TorchScript JIT archive
#             model = torch.jit.load(local_model_path, map_location="cpu").eval()
#             state_dict = None

#         except RuntimeError:
#             # If the above fails, load it as a state dictionary
#             state_dict = torch.load(local_model_path, map_location="cpu")

#         design_details = {
#             "trainer": 'MaPLe',
#             "vision_depth": 0,
#             "language_depth": 0,
#             "vision_ctx": 0,
#             "language_ctx": 0,
#             "maple_length": args.num_cont
#         }
        
#         # Build the model using the loaded state_dict or the JIT model's state_dict
#         model = clip.build_model(state_dict or model.state_dict(), design_details)

#         return model
    
#     CLIP_model = load_clip_to_cpu(args)
#     CLIP_model.float()

#     # Prepare input text for the model
#     if args.text_type == "class_names":
#         input_text = class_names
#     elif args.text_type == "class_names_with_context":
#         input_text = class_names_with_context
#     elif args.text_type == "class_descriptor":
#         input_text = class_descriptor

#     # Generate the model and load the pre-trained weights
#     model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
#     model = torch.nn.DataParallel(model).cuda()

#     print(model)
#     # for name, layer in model.module.clip_model.visual.named_modules():
#     #  layer.register_forward_hook(hook_fn)

  
#     #shufflenetww = torch.load('shufflnetw1_Ourmodel/1/Test_model.t7')
#     #efficientvitww = torch.load('efficientvitw2_Ourmodel/1/Test_model.t7')
#     ourmodel = torch.load('./log/fold1/Fullmodelbest/DFEW-2501041852GRUoursmodel-set1-model_best.pth')#Ours
#     #ourmodel = torch.load('./log/fold1/baseline/withaumaple/DFEW-2501051345firstconfgofAdap-set1-model_best.pth')#Baseline
#     #ourmodel = torch.load('./log/fold1/withoutTDA(sharedadapters/DFEW-2412280443withoutTDA-set1-model_best.pth')#Baseline+Sha
  
#     model.load_state_dict(ourmodel['state_dict'],strict=True) 
#     names = {
      
#         #'conv1': model.module.clip_model.visual.conv1,
#         'target_layer' : model.module.image_encoder.transformer.resblocks[-1].ln_1
#         #'conv1': model.module.image_encoder.ln_post,
#         #'attention_layer': model.module.image_encoder.transformer.resblocks[-1].attn
    
        
# }  
    


#     # transform = torchvision.transforms.Compose([
#     #      torchvision.transforms.ToPILImage(),
#     #     torchvision.transforms.Resize((224, 224)),
#     #     torchvision.transforms.ToTensor(),
       
#     # ])
    
#     transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.Resize((224, 224)),  # Resize to 224x224
#     torchvision.transforms.Lambda(lambda x: x[:, :, ::-1] if isinstance(x, np.ndarray) else x),  # Reverse channels if needed
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
# ])

#     torch.backends.cudnn.enabled = False
    
#     ## select part of test data to gen
#     for p in glob.glob('./test/*.jpg'):
#         for name,target_layer in names.items():
     
#             target_layer.register_forward_hook(hook_fn)
#             cam = methods['gradcam'](model=model,
#                             target_layers=[target_layer],
#                             use_cuda=True, reshape_transform=reshape_transform)
            
            

#             rgb_img = cv2.imread(p, 1)[:, :, ::-1].copy()
#             input_tensor = transform(rgb_img)
#             input_tensor = input_tensor.unsqueeze(0)
#             #input_tensor = input_tensor.unsqueeze(1)
#             input_tensor = input_tensor.to(device)
#             print('okkkkkkkkkk',input_tensor.shape)
#             print("Shape after patch embedding:", model.module.image_encoder.conv1(input_tensor).shape)
#             batch_size, embedding_dim, height, width = model.module.image_encoder.conv1(input_tensor).shape
#             flattened_input = model.module.image_encoder.conv1(input_tensor).permute(0, 2, 3, 1).reshape(batch_size, height * width, embedding_dim)
#             print("After flattening:", flattened_input.shape)
#             for i, block in enumerate(model.module.image_encoder.transformer.resblocks):
#                 x1 = block.ln_1(flattened_input)  # Apply LayerNorm
#                 print(f"After Block {i} ln_1: {flattened_input.shape}")
                
#                 if i == len(model.module.image_encoder.transformer.resblocks) - 2:
#                     print(f"Target layer output at Block {i}, ln_1: {x1.shape}")
#                     break

#                 xattn = block.attn(flattened_input, flattened_input, flattened_input)  # Apply attention
#                 print(f"After Block {i} attn: {flattened_input.shape}")

#                 x2 = block.ln_2(flattened_input)  # Apply LayerNorm
#                 print(f"After Block {i} ln_2: {flattened_input.shape}")

#                 xmlp = block.mlp(flattened_input)  # Apply MLP
#                 print(f"After Block {i} mlp: {flattened_input.shape}")
               


#             print(f"Input tensor is on device: {input_tensor.device}") 
#             try:
#                 # Generate Grad-CAM heatmap
#                 grayscale_cam = cam(input_tensor=input_tensor)
#                 grayscale_cam = grayscale_cam[0, :]  # Extract the first heatmap from the batch

#                 # Normalize heatmap for visualization
#                 grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())

#                 # Resize the heatmap to match the original image dimensions
#                 resized_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

#                 # Apply a colormap to the heatmap
#                 heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)

#                 # Overlay the heatmap on the original image
#                 cam_image = cv2.addWeighted(rgb_img, 0.7, heatmap, 0.3, 0)

#                 # Save or display the resulting image
#                 output_path = f'./Attentionmap/{os.path.basename(p)}_{name}_cam.jpg'
#                 cv2.imwrite(output_path, cam_image)
#                 print(f"Heatmap saved at {output_path}")

#             except ValueError as e:
#                 print(f"Grad-CAM error: {e}")




#             # grayscale_cam = cam(input_tensor=input_tensor)
#             # grayscale_cam = grayscale_cam[0, :]
#             # #grayscale_cam = np.max(grayscale_cam)
#             # # Normalize the heatmap
#             # grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())

#             # # Resize the heatmap to match the image size
#             # resized_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

#             # # Apply colormap to the resized heatmap
#             # heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)

#             # # Overlay the heatmap on the original image
#             # cam_image = cv2.addWeighted(rgb_img, 0.7, heatmap, 0.3, 0)

#             # cv2.imwrite(f'./Attentionmap/{os.path.basename(p)}_{name}_cam1.jpg', cam_image)




# import os
# import glob
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from models import FERPA
# import torchvision
# from torchvision import models
# import argparse
# import os
# import torch
# from dataloader.video_transform import *
# import torch.nn as nn
# import torch.nn.parallel
# import math
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# from thop import profile
# import torch.utils.data.distributed
# from models.FERPA import GenerateModel
# from fmix import  sample_mask
# import matplotlib
# matplotlib.use('Agg')
# from timm.scheduler import CosineLRScheduler
# import matplotlib.pyplot as plt
# import numpy as np
# import itertools
# import datetime
# from dataloader.video_dataloader import train_data_loader, test_data_loader
# from sklearn.metrics import confusion_matrix
# import tqdm
# from models.clip1 import clip
# import warnings
# import torchvision.transforms
# from FERPAmainmxp import RecorderMeter
# warnings.filterwarnings("ignore", category=UserWarning)
# from models.Text import *

# import random



# from pytorch_grad_cam import GradCAM, \
#                              ScoreCAM, \
#                              GradCAMPlusPlus, \
#                              AblationCAM, \
#                              XGradCAM, \
#                              EigenCAM, \
#                              EigenGradCAM

# from pytorch_grad_cam import GuidedBackpropReLUModel
# from pytorch_grad_cam.utils.image import show_cam_on_image, \
#                                          deprocess_image, \
#                                          preprocess_image

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str)
# parser.add_argument('--workers', type=int, default=0)
# parser.add_argument('--epochs', type=int, default=30)
# parser.add_argument('--batch-size', type=int, default=8)


# parser.add_argument('--lr-adapter', type=float, default=2e-4)
# parser.add_argument('--lr-prompt-learner', type=float, default=3e-5)

# parser.add_argument('--weight-decay', type=float, default=0.01)
# parser.add_argument('--warmup_epochs', type=int, default=2, help='number of warmup epochs.')
# parser.add_argument('--model-name', type=str, default="ViT-B/16")
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--print-freq', type=int, default=10)
# parser.add_argument('--milestones', nargs='+', type=int)
# parser.add_argument('--context-number', type=int, default=2)
# parser.add_argument('--num-cont', type=int, default=2)
# parser.add_argument('--init-text', type=str, default='False')
# parser.add_argument('--resume', type=str, default='False')
# parser.add_argument('--prompt-depth', type=int, default=9)
# parser.add_argument('--class-token-position', type=str, default="end")
# parser.add_argument('--class-specific-contexts', type=str, default='True')
# parser.add_argument('--load_and_tune_prompt_learner', type=str ,default='True')
# parser.add_argument('--text-type', type=str)
# parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--exper-name', type=str)
# parser.add_argument('--temporal-layers', type=int, default=1)
# args = parser.parse_args()

# # Define the RecorderMeter dummy class
# class RecorderMeter:
#     def __init__(self, *args, **kwargs):
#         pass

# def load_clip_to_cpu(args):
#     backbone_name = args.model_name
    
#     # Set the local path where you've saved the downloaded model
#     local_model_path = "F:\KmuProj2\Code4\models\ViT-B-16.pt"  # Update this path accordingly

#     try:
#         # Try to load the model as a TorchScript JIT archive
#         model = torch.jit.load(local_model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         # If the above fails, load it as a state dictionary
#         state_dict = torch.load(local_model_path, map_location="cpu")

#     design_details = {
#         "trainer": 'MaPLe',
#         "vision_depth": 0,
#         "language_depth": 0,
#         "vision_ctx": 0,
#         "language_ctx": 0,
#         "maple_length": args.num_cont
#     }
    
#     # Build the model using the loaded state_dict or the JIT model's state_dict
#     model = clip.build_model(state_dict or model.state_dict(), design_details)

#     return model

# if args.dataset == "FERV39K" or args.dataset == "DFEW":
#     number_class = 7
#     class_names = class_names_7
#     class_names_with_context = class_names_with_context_7
#     class_descriptor = class_descriptor_7


# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Wrap model with DataParallel


# # Modified code for visualizing attention maps
# if __name__ == '__main__':
#     os.makedirs('Code4/Attentionmap', exist_ok=True)

#     # Define CAM methods
#     methods = {
#         "gradcam": GradCAM,
#         "scorecam": ScoreCAM,
#         "gradcam++": GradCAMPlusPlus,
#         "ablationcam": AblationCAM,
#         "xgradcam": XGradCAM,
#         "eigencam": EigenCAM,
#         "eigengradcam": EigenGradCAM,
#     }

#     # Load the pre-trained CLIP model
#     CLIP_model = load_clip_to_cpu(args)
#     CLIP_model.float()

#     # Prepare input text for the model
#     if args.text_type == "class_names":
#         input_text = class_names
#     elif args.text_type == "class_names_with_context":
#         input_text = class_names_with_context
#     elif args.text_type == "class_descriptor":
#         input_text = class_descriptor

#     # Generate the model and load the pre-trained weights
#     model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
#     model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=1).to(device)

#     print(f"Using {torch.cuda.device_count()} GPUs.")
   
  
#     #print(model)

#     # Load pre-trained weights
#     ourmodel = torch.load('./log/fold1/Fullmodelbest/DFEW-2501041852GRUoursmodel-set1-model_best.pth')
#     model.load_state_dict(ourmodel['state_dict'], strict=True)

#     # Define the target layer for attention visualization
#     names = {
#         'last_transformer_block': model.module.image_encoder.transformer.resblocks[-1].attn,
#     }

#     for name, param in model.named_parameters():
#      print(f"{name} is on {param.device}")

#     transform = torchvision.transforms.Compose([
#         GroupResize(224),  # Works for a group of images
#         Stack(),
#         ToTorchFormatTensor()
#     ])

#     # Select a group of test data (e.g., frames in a folder)
#     img_paths = sorted(glob.glob('./DFEWdataset/Clip/clip_224x224/00001/*.jpg'))
#     img_group = [Image.open(img_path).convert('RGB') for img_path in img_paths]  # List of PIL images

#     # Apply transformations
#     input_tensor = transform(img_group)  # GroupResize and Stack applied

#     # Debug: Check transform output
#     print(f"Transform output type: {type(input_tensor)}")
#     print(f"Transform output shape: {input_tensor.shape}")

#     # Reshape into (T, C, H, W) if needed
#     if len(input_tensor.shape) == 3:  # Shape is (T*H, W, C)
#         num_frames = len(img_group)
#         input_tensor = input_tensor.view(num_frames, 3, 224, 224).to(device)  # Assume (T, C, H, W)
#     elif len(input_tensor.shape) == 4:
#         input_tensor = input_tensor  # Already in correct shape
#     else:
#         raise TypeError(f"Unexpected transform output shape: {input_tensor.shape}. Expected 4D tensor.")

#     # Add batch dimension
#     input_tensor = input_tensor.unsqueeze(0).to(device) # Shape becomes (B, T, C, H, W)

#     for name, target_layer in names.items():
#         cam = methods['xgradcam'](model=model, target_layers=[target_layer], use_cuda=True)

#         # Generate CAM
#         grayscale_cams = cam(input_tensor=input_tensor)
#         for i, grayscale_cam in enumerate(grayscale_cams):
#             # Normalize heatmap
#             grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())

#             # Resize the heatmap to match the image size
#             resized_cam = cv2.resize(grayscale_cam, (img_group[i].width, img_group[i].height))

#             # Create heatmap
#             heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
#             original_image = np.array(img_group[i])[:, :, ::-1]  # Convert RGB to BGR
#             cam_image = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)

#             # Save result
#             output_path = f'./Attentionmap/{os.path.basename(img_paths[i])}_{name}_cam.jpg'
#             cv2.imwrite(output_path, cam_image)
#             print(f"Saved attention map to: {output_path}")


