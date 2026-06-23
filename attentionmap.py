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
        local_model_path = "F:\..\..\models\ViT-B-16.pt"  # Update this path accordingly

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


    # set the best model path
    ourmodel = torch.load('./bestmodle.pth')#Ours
  
    model.load_state_dict(ourmodel['state_dict'],strict=True) 


    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Assuming 16-frame video input

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

