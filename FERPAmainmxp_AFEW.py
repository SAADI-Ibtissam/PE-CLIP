

# ###################################################################
#                   #CrossEntropyLoss
# ###################################################################
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import math
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from thop import profile
import torch.utils.data.distributed
from models.FERPA import GenerateModel
from fmix import  sample_mask
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
warnings.filterwarnings("ignore", category=UserWarning)
from models.Text import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch-size', type=int, default=8)


parser.add_argument('--lr-adapter', type=float, default=1e-4)
parser.add_argument('--lr-prompt-learner', type=float, default=3e-4)

parser.add_argument('--weight-decay', type=float, default=1e-2)
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


args = parser.parse_args()


# For reproductibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("GPU Name:", torch.cuda.get_device_name(0))

NUM_CLASSES = 7

now = datetime.datetime.now()
time_str = now.strftime("%y%m%d%H%M")
time_str = time_str + args.exper_name

print('************************')
print('************************')

if args.dataset == "AFEW" :
    number_class = 7
    class_names = class_names_7
    class_names_with_context = class_names_with_context_7
    class_descriptor = class_descriptor_7


# def load_clip_to_cpu(args):
#     backbone_name = args.model_name
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url, root='./models')

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")
#     design_details = {"trainer": 'MaPLe',
#                       "vision_depth": 0,
#                       "language_depth": 0, "vision_ctx": 0,
#                       "language_ctx": 0,
#                       "maple_length": args.num_cont}
#     model = clip.build_model(state_dict or model.state_dict(), design_details)

#     return model

def load_clip_to_cpu(args):
    backbone_name = args.model_name
    
    # Set the local path where you've saved the downloaded model
    #local_model_path = "D:\KmuProj2\Code4\models\ViT-B-16.pt"
    local_model_path = "models/ViT-B-16.pt"

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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

#################################
def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.7)
    data = lam*data + (1-lam)*shuffled_data
    targets = (target, shuffled_target, lam)

    return data, targets

def mixup_loss_fn(criterion, preds, targets):
    targets1, targets2, lam = targets
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data = data[indices].to(device)
    shuffled_targets = targets[indices].to(device)
    x1 = torch.from_numpy(mask).to(device)*data.to(device)
    x2 = torch.from_numpy(1-mask).to(device)*shuffled_data.to(device)
    targets=(targets, shuffled_targets, lam)
    
    return (x1+x2), targets


def fmix_loss_fn(criterion, outputs, targets):
    y_a, y_b, lam = targets
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

def mixup_or_fmix(data, targets, alpha_mixup=0.4, alpha_fmix=1.0, decay_power=3.0, shape=(224, 224), prob_p=0.4, prob_f=0.7):
    # Randomly choose between Mixup and Fmix
    p = random.random()
    if p < prob_p:
        # Apply Mixup
        mixed_data, mixed_targets = mixup(data, targets, alpha=alpha_mixup)
        method = 'mixup'
   
    #elif p < prob_p + prob_f:
    elif p >= prob_p and p < prob_f:
        # Apply Fmix
        mixed_data, mixed_targets = fmix(data, targets, alpha=alpha_fmix, decay_power=decay_power, shape=shape)
        method = 'fmix'
   
    else: 
        mixed_data, mixed_targets = data, targets
        method = 'none'
    return mixed_data, mixed_targets, method


# def mixup(data, target, alpha):
#     indices = torch.randperm(data.size(0))
#     shuffled_data = data[indices]
#     shuffled_target = target[indices]

#     lam = np.clip(np.random.beta(alpha, alpha),0.3,0.7)
#     data = lam*data + (1-lam)*shuffled_data
#     targets = (target, shuffled_target, lam)

#     return data, targets

# def mixup(x, y, alpha=1.0, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).to(device)
#     else:
#         index = torch.randperm(batch_size)

#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




def print_model_parameters(model):
    print(f"{'Parameter Name':<40} {'Shape':<30} {'Requires Grad'}")
    print("="*80)
    for name, param in model.named_parameters():
        print(f"{name:<40} {str(param.shape):<30} {param.requires_grad}")

def main(set):
    #data_set = set + 1
    data_set=1
    
    if args.dataset == "AFEW":
        print("*********** AFEW Dataset ***********")
        log_txt_path = './log/' + 'AFEW-' + time_str + '-log.txt'
        log_curve_path = './log/' + 'AFEW-' + time_str + '-log.png'
        log_confusion_matrix_path = './log/' + 'AFEW-' + time_str + '-cn.png'
        checkpoint_path = './checkpoint/' + 'AFEW-' + time_str + '-model.pth'
        best_checkpoint_path = './checkpoint/' + 'AFEW-' + time_str + '-model_best.pth'
        epoch_checkpoint_path = './checkpoint/' + 'AFEW-' + time_str + '-epoch_{}.pth'
        train_annotation_file_path = "./DFEWdataset/Annotation/AFEW_train.txt"
        test_annotation_file_path = "./DFEWdataset/Annotation/AFEW_val.txt"
        pretrained_checkpoint_path = 'log/fold1/Fullmodelbest/PE-CLIP_BestModel.pth'

    start_epoch = 0
    best_acc = 0
    best_epoch = 0
    recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)

    # create model and load pre_trained parameters
    #CLIP_model, _ = clip.load("ViT-B/32", device='cpu')
    CLIP_model = load_clip_to_cpu(args)
    CLIP_model.float()
    
    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor

    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    #pre_trained_dict = torch.load(pretrained_checkpoint_path, )['state_dict']
    ckpt = torch.load(pretrained_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model = torch.nn.DataParallel(model).to(device)

    # Make all parameters non-trainable first


    for param in model.parameters():
        param.requires_grad = False

    # Now, iterate over the CLIP model's parameters and set the adapter ones as trainable
    for name, param in model.module.clip_model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
            print(f"Adapter parameter set to trainable: {name}")
    
    # Unfreeze prompt learner parameters
    for name, param in model.named_parameters():
        if 'prompt_learner' in name:
            param.requires_grad = True
            print(f"Prompt learner parameter set to trainable: {name}")

        # if 'text_refinement' in name:
        #     param.requires_grad = True
        #     print(f"Text refinment parameter set to trainable: {name}")

        # if 'image_alignment_layer' in name:
        #     param.requires_grad = True
        #     print(f"image_alignment_layer parameter set to trainable: {name}")

        # if 'text_alignment_layer' in name:
        #     param.requires_grad = True
        #     print(f"text_alignment_layer parameter set to trainable: {name}")
        
        
    
    # Optionally, you can print all the parameters to verify
    # for name, param in model.module.clip_model.named_parameters():
    #     print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    # Initialize counters
    total_params = 0
    total_trainable_params = 0
    trainable_adapter_params = 0
    trainable_prompt_params = 0

    # Loop through all parameters
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            total_trainable_params += num_params
            
            if 'adapter'  in name:
                trainable_adapter_params += num_params
            elif 'prompt_learner' in name:
                trainable_prompt_params += num_params

   

    # Print the numbers
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Adapter Parameters: {trainable_adapter_params:,}")
    print(f"Trainable Prompt Parameters: {trainable_prompt_params:,}")
    print(f"Total Trainable Parameters: {total_trainable_params:,}")
    
    #print_model_parameters(model)
    
    with open(log_txt_path, 'a') as f:
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable Adapter Parameters: {trainable_adapter_params:,}\n")
        f.write(f"Trainable Prompt Parameters: {trainable_prompt_params:,}\n")
        f.write(f"Trainable parameters: {total_trainable_params:,}\n")
   

        

    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')
    
    # define loss function (criterion)
    #criterion = nn.CrossEntropyLoss().to(device)
    #alpha = torch.tensor([1, 2, 3, 1, 1, 10, 2], dtype=torch.float32)  # Increased weight for Class 6
    criterion = nn.CrossEntropyLoss().to(device)
    
    # define optimizer
    optimizer = torch.optim.AdamW([{"params": [p for n, p in model.module.clip_model.named_parameters() if 'adapter' in n], 
                                    "lr": args.lr_adapter, "weight_decay": args.weight_decay},
                                 {"params": model.module.prompt_learner.parameters(), "lr": args.lr_prompt_learner, 
                                  "weight_decay": args.weight_decay},]
                                 
                                 )
    
  
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) 

    # scheduler = CosineLRScheduler(optimizer, t_initial=30, warmup_t=3,
    #                                 warmup_lr_init=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=10,                 # Initial cycle length of 10 epochs
#     T_mult=2,               # Doubling cycle length after each restart
#     eta_min=2e-5            # Minimum learning rate to avoid overshooting
# )
   
    # scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=30,
    #         t_mul=getattr(args, 'lr_cycle_mul', 1.),
    #         lr_min=3e-6,
    #         decay_rate=0.5,
    #         warmup_lr_init=3e-5,
    #         warmup_t=3,
    #         cycle_limit=getattr(args, 'lr_cycle_limit', 1),
    #         t_in_epochs=True,
    #         noise_pct=getattr(args, 'lr_noise_pct', 0.5),
    #         noise_std=getattr(args, 'lr_noise_std', 1.),
    #         noise_seed=getattr(args, 'seed', 42),
    #     )
        


    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=2e-5)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                   milestones=args.milestones,
    #                                                   gamma=0.1)
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(
                                            optimizer,
                                            step_size=8,  # drops at epochs 8,16,24
                                            gamma=0.5     # gentler than 0.1 (better for short runs)
)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
    #                                                       total_iters=500)

   
        
    cudnn.benchmark = True
    
     # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, weights_only=False)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint.get('best_epoch', start_epoch)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and scheduler is not None:
              scheduler.load_state_dict(checkpoint['scheduler'])
              print("Loaded scheduler state from checkpoint")
            recorder = checkpoint['recorder']

            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Data loading code
    train_data = train_data_loader(list_file=train_annotation_file_path,
                                   num_segments=16,
                                   duration=1,
                                   image_size=224,
                                   args=args)
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=224)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=2.5e-4,             # Peak learning rate
#     steps_per_epoch=len(train_loader),
#     epochs=30,                 # Total number of epochs
#     pct_start=0.3,             # 30% of the cycle increasing the learning rate
#     anneal_strategy='cos',     # Cosine decay after reaching max_lr
#     div_factor=25,             # Initial LR = max_lr / div_factor
#     final_div_factor=10        # Final LR = max_lr / final_div_factor at the end of training
# )
    
   
    
  # define scheduler
    # def lr_func(step):
    #   epoch = step / len(train_loader)
    #   if epoch < args.warmup_epochs:
    #     return epoch / args.warmup_epochs
    #   else:
    #     return 0.5 + 0.5 * math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)



    for epoch in range(start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        current_learning_rate1 = optimizer.state_dict()['param_groups'][1]['lr']
        #current_learning_rate2 = optimizer.state_dict()['param_groups'][2]['lr']
        #current_learning_rate3 = optimizer.state_dict()['param_groups'][3]['lr']
        #current_learning_rate4 = optimizer.state_dict()['param_groups'][4]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            # f.write('Current learning rate: ' + str(current_learning_rate) + '\n')
            # print('Current learning rate: ', current_learning_rate)  
            f.write('Current learning rate: ' + str(current_learning_rate) + ' ' + str(current_learning_rate1)  +  '\n')
            print('Current learning rate: ', current_learning_rate, current_learning_rate1,)
            
            #With combine loinear
            # f.write('Current learning rate: ' + str(current_learning_rate) + ' ' + str(current_learning_rate1)  +  ' ' + str(current_learning_rate2)   +  ' ' + str(current_learning_rate3)  +  ' ' + str(current_learning_rate4) + '\n')
            # print('Current learning rate: ', current_learning_rate, current_learning_rate1 , current_learning_rate2, current_learning_rate3, current_learning_rate4, )          
            
       
            
        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args, log_txt_path)
        
        scheduler.step(epoch)

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        if is_best:

            best_acc = val_acc
            best_epoch = epoch + 1 

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'best_epoch': best_epoch,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 
                         'recorder': recorder}, is_best,
                        checkpoint_path,
                        best_checkpoint_path)

        # Save checkpoint after every epoch
        epoch_checkpoint = epoch_checkpoint_path.format(epoch + 1)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'best_epoch': best_epoch,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 
                         'recorder': recorder},
                         is_best=False,
                         checkpoint_path=epoch_checkpoint,
                         best_checkpoint_path=None)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')
            f.write('The best epoch: ' + str(best_epoch) + 's' + '\n')

    # Print the best epoch and accuracy
    print(f'Best validation accuracy of {best_acc:.3f} was achieved at epoch {best_epoch}.')
    with open(log_txt_path, 'a') as f:
        f.write(f'Best validation accuracy of {best_acc:.3f} was achieved at epoch {best_epoch}.\n')   

    uar, war = computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set)
    
    return uar, war

def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    print("start training")
    losses = AverageMeter('Loss', ':.7f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)
    
    #cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    #mixup = MixUp(num_classes=NUM_CLASSES)
    #cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        mixed_data, mixed_targets, method = mixup_or_fmix(images, target)

        # images, targets_a, targets_b, lam = mixup(images, target)
        # images, targets_a, targets_b = map(torch.autograd.Variable, (images, targets_a, targets_b))

        output = model(mixed_data)

        if method == 'mixup':
           loss = mixup_loss_fn(criterion, output, mixed_targets)
           #loss = criterion(output, mixed_targets[0]) * mixed_targets[2] + criterion(output, mixed_targets[1]) * (1. - mixed_targets[2])
        elif method == 'fmix':
           loss = fmix_loss_fn(criterion, output, mixed_targets)

           #loss = criterion(output, mixed_targets[0]) * mixed_targets[2] + criterion(output, mixed_targets[1]) * (1. - mixed_targets[2])
        else:
            loss = criterion(output, target)


      
      
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    progress.display(len(train_loader) - 1)
    print(' * Epoch: [{}], Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch, losses.avg, top1.avg))
    with open(log_txt_path, 'a') as f:
        f.write(' * Epoch: [{}], Loss: {:.4f}, Accuracy: {:.3f}\n'.format(epoch, losses.avg, top1.avg))

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.7f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output= model(images)

            # Ensure target tensor has the correct shape
            #target = target.view(-1)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader) - 1)
        print(' * Test, Loss: {:.4f}, Accuracy: {:.3f}'.format(losses.avg, top1.avg))
        with open(log_txt_path, 'a') as f:
            f.write(' * Test, Loss: {:.4f}, Accuracy: {:.3f}\n'.format(losses.avg, top1.avg))


        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
                f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')

        computer_uar_war_val(val_loader, model, log_txt_path)

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set):
    pre_trained_dict = torch.load(best_checkpoint_path, weights_only=False)['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(val_loader)):
            images = images.to(device)
            target = target.to(device)

            output= model(images)        

            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)

    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()
        
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    if args.dataset == "AFEW":
        title_ = "Confusion Matrix on AFEW"
    

    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()
    
    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')        
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('************************' + '\n')
    
    return uar, war

def computer_uar_war_val(val_loader, model, log_txt_path):
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(val_loader)):
            images = images.to(device)
            target = target.to(device)

            #output, text_features, refined_text_features  = model(images)   

            output= model(images)        

            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset) 
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()
 
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    with open(log_txt_path, 'a') as f:
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write("UAR: %0.2f\n" % uar)
        f.write("WAR: %0.2f\n" % war)

if __name__ == '__main__':
    UAR = 0.0
    WAR = 0.0

    if args.dataset == "AFEW":
        all_fold = 1
   

    for set in range(all_fold):
        uar, war = main(set)
        UAR += float(uar)
        WAR += float(war)
        
    print('********* Final Results *********')   
    print("UAR: %0.2f" % (UAR / all_fold))
    print("WAR: %0.2f" % (WAR / all_fold))
    print('*********************************')



