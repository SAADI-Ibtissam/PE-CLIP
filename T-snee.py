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
from sklearn.manifold import TSNE
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
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=8)


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


args = parser.parse_args()


# For reproductibility
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(RANDOM_SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

NUM_CLASSES = 7

now = datetime.datetime.now()
time_str = now.strftime("%y%m%d%H%M")
time_str = time_str + args.exper_name

print('************************')
print('************************')

if args.dataset == "FERV39K" or args.dataset == "DFEW":
    number_class = 7
    class_names = class_names_7
    class_names_with_context = class_names_with_context_7
    class_descriptor = class_descriptor_7


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


def main(set):
    #data_set = set + 1
    data_set=1
    
    if args.dataset == "FERV39K":
        print("*********** FERV39K Dataset ***********")
        log_txt_path = './log/' + 'FER39K-' + time_str + '-log.txt'
        log_curve_path = './log/' + 'FER39K-' + time_str + '-log.png'
        log_confusion_matrix_path = './log/' + 'FER39K-' + time_str + '-cn.png'
        checkpoint_path = './checkpoint/' + 'FER39K-' + time_str + '-model.pth'
        best_checkpoint_path = './checkpoint/' + 'FER39K-' + time_str + '-model_best.pth'
        epoch_checkpoint_path = './checkpoint/' + 'FER39K-' + time_str + '-epoch_{}.pth'
        train_annotation_file_path = "./DFEWdataset/Annotation/FERV39K_train.txt"
        test_annotation_file_path = "./DFEWdataset/Annotation/FERV39K_test.txt"
    
    elif args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log.txt'
        log_curve_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log.png'
        log_confusion_matrix_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-cn.png'
        checkpoint_path = "checkpoint/" + 'DFEW-' + time_str + '-set' + str(data_set) + '-model.pth'
        best_checkpoint_path = "checkpoint/" + 'DFEW-' + time_str + '-set' + str(data_set) + '-model_best.pth'
        epoch_checkpoint_path = "checkpoint/" + 'DFEW-' + time_str + '-set' + str(data_set) + '-epoch_{}.pth'
        train_annotation_file_path = "DFEWdataset/Annotation/DFEW_set_" + str(data_set) + "_train.txt"
        print(train_annotation_file_path)
        test_annotation_file_path = "DFEWdataset/Annotation/DFEW_set_" + str(data_set) + "_test.txt"
        print(test_annotation_file_path)

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
    model = torch.nn.DataParallel(model).cuda()

    ourmodel = torch.load('./log/fold1/Fullmodelbest/DFEW-2501041852GRUoursmodel-set1-model_best.pth')#Ours
    #ourmodel = torch.load('./log/fold1/baseline/withaumaple/DFEW-2501051345firstconfgofAdap-set1-model_best.pth')#Baseline
    #ourmodel = torch.load('./log/fold1/withoutTDA(sharedadapters/DFEW-2412280443withoutTDA-set1-model_best.pth')#Baseline+Sha
    
    model.load_state_dict(ourmodel['state_dict'],strict=True) 
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # define optimizer
    optimizer = torch.optim.AdamW([{"params": [p for n, p in model.module.clip_model.named_parameters() if 'adapter' in n], 
                                    "lr": args.lr_adapter, "weight_decay": args.weight_decay},
                                 {"params": model.module.prompt_learner.parameters(), "lr": args.lr_prompt_learner, 
                                  "weight_decay": args.weight_decay},]
                                 
                                 )
    


    scheduler = CosineLRScheduler(optimizer, t_initial=30, warmup_t=3,
                                   warmup_lr_init=2e-5)

        
    cudnn.benchmark = True
    
   
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=224)


    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    

   
    

    for epoch in range(start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        current_learning_rate1 = optimizer.state_dict()['param_groups'][1]['lr']
     
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
          
            f.write('Current learning rate: ' + str(current_learning_rate) + ' ' + str(current_learning_rate1)  +  '\n')
            print('Current learning rate: ', current_learning_rate, current_learning_rate1,)
            

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
       
    uar, war = computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set)
    
    return uar, war




def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.7f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    # switch to evaluate mode
    model.eval()
    total_inference_time = 0  # Track total inference time
    num_samples = 0           # Track number of sequences

    # Initialize features and targets as lists
    features = []  # This will store all the features
    targets = []   # This will store all the target labels

    with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

              
                # compute output
                start_time = time.time()
        
                output = model(images)
                end_time = time.time()

                # Compute elapsed time
                inference_time = end_time - start_time
                total_inference_time += inference_time
                num_samples += images.size(0)
                
                # Track GPU Memory Usage
                allocated_memory = torch.cuda.memory_allocated() / 1e6  # Convert bytes to MB
                reserved_memory = torch.cuda.memory_reserved() / 1e6  # Convert bytes to MB

                print(f"Batch {i}: Allocated Memory: {allocated_memory:.2f} MB | Reserved Memory: {reserved_memory:.2f} MB")


                # Convert the output tensor to NumPy
                current_outputs = output.cpu().numpy()

                # Append the current batch's features and targets to the lists
                features.append(current_outputs)
                targets.append(target.cpu().numpy())  # Convert target to NumPy and append

                # Compute loss
                loss = criterion(output, target)

                # Measure accuracy and record loss
                acc1, _ = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

                if i % args.print_freq == 0:
                    progress.display(i)
            
             # Compute average inference time per sequence
            avg_inference_time = total_inference_time / num_samples
            print(f"Average Inference Time per Video Sequence: {avg_inference_time:.4f} seconds")


            # Concatenate features and targets into NumPy arrays after the loop
            features = np.concatenate(features, axis=0)  # Combine all feature batches
            targets = np.concatenate(targets, axis=0)    # Combine all target batches
           
            class_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=5)  # Keep only the top 50 components
            features_reduced = pca.fit_transform(features)

            # Perform t-SNE on the collected features
            tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate=250, perplexity=30, n_iter=2000).fit_transform(features_reduced)

            def scale_to_01_range(x):
                value_range = (np.max(x) - np.min(x))
                starts_from_zero = x - np.min(x)
                return starts_from_zero / value_range

            # Scale t-SNE results to [0, 1] range
            tx = scale_to_01_range(tsne[:, 0])
            ty = scale_to_01_range(tsne[:, 1])
            print(f"Features Shape: {features.shape}")
            print(f"Targets Shape: {targets.shape}")
            print(f"t-SNE X-coordinates: {tx[:5]}")  # Print first 5 values
            print(f"t-SNE Y-coordinates: {ty[:5]}")  # Print first 5 values

            # Plot t-SNE results
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)

            # Color list for visualization
            #colors = ['red', 'blue', 'green', 'brown', 'yellow', 'orange', 'purple']
            colors = ['darkorange','crimson','seagreen','lightpink', 'yellow', 'royalblue', 'rebeccapurple']

            # Plot each class
            unique_classes = np.unique(targets)  # Get unique class labels

            for class_id in unique_classes:
                indices = np.where(targets == class_id)
                current_tx = tx[indices]
                current_ty = ty[indices]

                # Use a color for each class ID
                color = colors[int(class_id) % len(colors)]  # Ensure valid indexing

                ax.scatter(
                    current_tx, current_ty,
                    label=f"{class_names[class_id]}",
                    color=color,
                    alpha=1.0,
                    s=15
                )

            ax.legend(loc='best')
            ax.set_title('t-SNE Visualization of Features (Baseline) fd1')
            plt.savefig('tsne_ourspcaf1test.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("t-SNE plot saved and displayed.")


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
    pre_trained_dict = torch.load(best_checkpoint_path)['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(val_loader)):
            images = images.cuda()
            target = target.cuda()

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

    if args.dataset == "FERV39K":
        title_ = "Confusion Matrix on FERV39k"
    elif args.dataset == "DFEW":
        title_ = "Confusion Matrix on DFEW fold " + str(data_set)

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
            images = images.cuda()
            target = target.cuda()

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

    if args.dataset == "FERV39K":
        all_fold = 1
    elif args.dataset == "DFEW":
        all_fold = 1

    for set in range(all_fold):
        uar, war = main(set)
        UAR += float(uar)
        WAR += float(war)
        
    print('********* Final Results *********')   
    print("UAR: %0.2f" % (UAR / all_fold))
    print("WAR: %0.2f" % (WAR / all_fold))
    print('*********************************')

