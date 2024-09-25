

"""Train CIFAR10 with PyTorch."""
import copy
import math
import os
import random
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import grad
from decouple import config
from src.modeling import ImageClassifier, ImageEncoder
from src.heads import get_classification_head
# from utils import cosine_lr
from sklearn.metrics import accuracy_score
from model import evaluate_model_freeze, evaluate_model_cam_ensemble_freeze
from einops import rearrange
from utils.utils import get_logger
from utils.cli_utils import *
import time
import wandb
from utilities import Paths, calculate_metrics, brier_score, calculate_auroc_multiclass

#seed = 113
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)



 

def generate_freezed_particles(mdl , num_ensemble, device):

    classification_head = get_classification_head()
    image_encoder = ImageEncoder(mdl)
    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()

    NET = NET.to(device)
    particles = []
    for i in range(num_ensemble):
            particles.append(copy.deepcopy(NET))

    print(f'number of individual models: {len(particles)}')  
    
    return particles  

def train_model_wrap_cifar(particles, trainloaders, valloader, noise_std, k, config):
    h_kernel = 0
    criterion = nn.CrossEntropyLoss()

    best_losses = [float('inf')] * len(particles)
    best_val_accuracy = [float('inf')] * len(particles)

    learning_rates = [0.001, 0.0007, 0.0005, 0.00025, 0.0008]

    optimizers = [optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr) for model, lr in zip(particles, learning_rates)]


    for epoch in range(int(config('num_epochs'))):
        
        accumulated_losses = [0.0] * len(particles)
        num_batches = len(next(iter(trainloaders)))

        for j,batches in enumerate(zip(*trainloaders)):
            inputs_list = [batch[0] for batch in batches]
            targets_list = [batch[1] for batch in batches]
            for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):
                imgs, labels = imgs.cuda(), lbls.cuda()

                optimizers[i].zero_grad()

                logits = model(imgs)

                loss = criterion(logits, labels)
                loss.backward()
                accumulated_losses[i] += loss.item()
            print(f'\rProcessing batch {j+1}/{num_batches}', end='')
            
            # h_kernel = update_gradiants(particles, h_kernel)

            for optimizer in optimizers:
                optimizer.step()
        print(" ")
        average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
        for i, avg_loss in enumerate(average_losses):
            print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

    
        with torch.no_grad():
            for i,model in enumerate(particles):

                correct = 0
                total = 0
                losses_eval, step2 = 0., 0.
                for img, lbls,_ in valloader:
                    img, label = img.cuda(), lbls.cuda()

                    logits = model(img)
                    loss_val = criterion(logits, label)
                    losses_eval += loss_val.item()
                    _, predicted = torch.max(logits, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    step2 += 1

                accuracy = correct / total
                loss_val_final = losses_eval / step2
                print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}, val_loss_{i}: {loss_val_final:.4f}')
                
                # 3. Save Models with Best Validation Loss
                model_idx = particles.index(model)
                if loss_val_final < best_losses[model_idx]:
                    best_losses[model_idx] = loss_val_final
                    best_val_accuracy[model_idx] = accuracy
                    best_epoch = epoch
                    best_model = copy.deepcopy(model.state_dict())

                    best_model_path = f"/media/tower2/DATA4/Afshar/results/saved_models/set/cam/best_model_{i}_series_{k}.pt"
                    torch.save(best_model, best_model_path)
                    print(f'Best model {i} at epoch {best_epoch} has been saved')

    with open(f"/media/tower2/DATA4/Afshar/results/saved_models/set/cam/best_val_accuracy_{k}.txt", "w") as file:
    # Write each accuracy value to the file, one value per line
        for i,accuracy in enumerate(best_val_accuracy):
            file.write(f"best val_acc for model {i} is {accuracy}\n")
    print('finished')        



#--------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
  

def compute_input_gradients(model, imgs):
    imgs.requires_grad = True
    logits = model(imgs)
    entropies = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    loss = entropies.mean(0)
    input_gradients = torch.autograd.grad(outputs=loss, inputs=imgs, create_graph=True)[0].detach()
    imgs.requires_grad = False
    model.zero_grad()
    return input_gradients, entropies, logits



def adapt_BTTA(particles, test_loader, device, config):
    seed = 2295
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if config('dataset_name').upper() == "CIFAR10":
        grad_threshold = 0.02

    elif config('dataset_name').upper() == "CIFAR100":
        grad_threshold = 0.08 # default is 0.07

    h_kernel = 0
    for model in particles:
        model.to(device)

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(config('lr')), momentum=0.9 ) 

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    for epoch in range(1):
        
        end = time.time()
        st_time = time.time()

        all_norms = []
        num_filtered_out = 0
        logits_test, targets_test = [], []
        norms_list, entropies_list, plpd_list = [], [], []
        for j, (imgs, lbls) in enumerate(test_loader):
            imgs, labels = imgs.to(device), lbls.to(device)
            logits = []

            for i in range(len(particles)):
                optimizers[i].zero_grad()


                input_grads, entropies, l = compute_input_gradients(particles[i], imgs)
                l2_norm = torch.norm(input_grads, p=2, dim=(1, 2, 3))

                logits.append(l)

                filter_ids_1 = torch.where((entropies < 50) & (l2_norm < grad_threshold))
                filtered_out_ids = torch.where((entropies < 0.5) & (l2_norm >= grad_threshold))
                num_filtered_out = filtered_out_ids[0].numel()

                entropys = entropies[filter_ids_1]

                x_prime = imgs[filter_ids_1]
                x_prime = x_prime.detach()

                patch_len=4

                resize_t = torchvision.transforms.Resize(((imgs.shape[-1]//patch_len)*patch_len,(imgs.shape[-1]//patch_len)*patch_len))
                resize_o = torchvision.transforms.Resize((imgs.shape[-1],imgs.shape[-1]))
                x_prime = resize_t(x_prime)
                x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
                perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
                x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
                x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
                x_prime = resize_o(x_prime)

                with torch.no_grad():
                    outputs_prime = model(x_prime)
                
                prob_outputs = l[filter_ids_1].softmax(1)
                prob_outputs_prime = outputs_prime.softmax(1)

                cls1 = prob_outputs.argmax(dim=1)

                plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
                plpd = plpd.reshape(-1)

                plpd_threshold = 0.2
                filter_ids_2 = torch.where(plpd > plpd_threshold)
                entropys = entropys[filter_ids_2]
             
                if len(entropys) !=0:
                    loss = entropys.mean(0)
                    loss.backward()
                    optimizers[i].step()
   
            logits = torch.stack(logits).mean(0)
            logits_test.append(logits.cpu().detach().numpy())
            targets_test.append(labels.cpu().detach().numpy())

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))   
            top1.update(acc1[0], logits.size(0))
            top5.update(acc5[0], logits.size(0))

            if (j+1) % int(config('wandb_interval')) == 0:
                progress.display(j)

            batch_time.update(time.time() - end)
            end = time.time()               
        en_time = time.time()
        print(f'time: {en_time - st_time}')
        print(f"acc1s are {top1.avg.item()}")
        print(f"acc5s are {top5.avg.item()}")

        logits_test = np.concatenate(logits_test, axis=0)
        targets_test = np.concatenate(targets_test, axis=0)

        lgits = torch.tensor(logits_test, dtype=torch.float32)
        targets = torch.tensor(targets_test, dtype=torch.long)
        preds = torch.argmax(lgits, dim=1)
        correct = (preds == targets).sum().item()
        ac = correct / targets.size(0)

        num_classes = int(config('num_class'))
        ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
        brier = brier_score(logits_test, targets_test, num_classes)
        AUROC = calculate_auroc_multiclass(logits_test, targets_test, num_classes)
        print(
            '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f, MCE = %.4f, Brier = %.5f, AUROC = %.4f' %
            (ac, ECE, MCE, brier, AUROC)
        )     

    
###############################################################################################################
###############################################################################################################

def validate(val_loader, model, device, mode='eval'):
    biased = False
    wandb_log = False

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    if biased:
        LL_AM = AverageMeter('LL Acc', ':6.2f')
        LS_AM = AverageMeter('LS Acc', ':6.2f')
        SL_AM = AverageMeter('SL Acc', ':6.2f')
        SS_AM = AverageMeter('SS Acc', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, LL_AM, LS_AM, SL_AM, SS_AM],
            prefix='Test: ')
        
    model.eval()

    with torch.no_grad():
        end = time.time()
        st_time = time.time()
        correct_count = [0,0,0,0]
        total_count = [1e-6,1e-6,1e-6,1e-6]
        logits_test, targets_test= [], []
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            images, target = images.to(device), target.to(device)
            if biased:
                if config('dataset_name').upper()=='Waterbirds':
                    place = dl[2]['place'].cuda()
                else:
                    place = dl[2].cuda()
                group = 2*target + place #0: landbird+land, 1: landbird+sea, 2: seabird+land, 3: seabird+sea
                
            # compute output
            if config('method').lower()=='deyo':
                output = adapt_model(images, i, target, flag=False, group=group)
            else:
                output = model(images)

            logits_test.append(output.cpu().detach().numpy())
            targets_test.append(target.cpu().detach().numpy())

            # measure accuracy and record loss
            if biased:
                TFtensor = (output.argmax(dim=1) == target)
                for group_idx in range(4):
                    correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                    total_count[group_idx] += len(TFtensor[group==group_idx])
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
                

            # '''
            if (i+1) % int(config('wandb_interval')) == 0:
                if biased:
                    LL = correct_count[0]/total_count[0]*100
                    LS = correct_count[1]/total_count[1]*100
                    SL = correct_count[2]/total_count[2]*100
                    SS = correct_count[3]/total_count[3]*100
                    LL_AM.update(LL, images.size(0))
                    LS_AM.update(LS, images.size(0))
                    SL_AM.update(SL, images.size(0))
                    SS_AM.update(SS, images.size(0))
                    if wandb_log:
                        wandb.log({f"{config('corruption')}/LL": LL,
                                   f"{config('corruption')}/LS": LS,
                                   f"{config('corruption')}/SL": SL,
                                   f"{config('corruption')}/SS": SS,
                                  })
                if wandb_log:
                    wandb.log({f"{config('corruption')}/top1": top1.avg,
                               f"{config('corruption')}/top5": top5.avg})
                
                progress.display(i)
            # '''
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            '''
            if (i+1) % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break
            '''
            
        ed_time = time.time()

    print(f"acc1s are {top1.avg.item()}")
    print(f"acc5s are {top5.avg.item()}")
    print(f'time: {ed_time - st_time}')

    logits_test = np.concatenate(logits_test, axis=0)
    targets_test = np.concatenate(targets_test, axis=0)

    logits = torch.tensor(logits_test, dtype=torch.float32)
    targets = torch.tensor(targets_test, dtype=torch.long)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    ac = correct / targets.size(0)


    num_classes = int(config('num_class'))
    ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
    brier = brier_score(logits_test, targets_test, num_classes)
    AUROC = calculate_auroc_multiclass(logits_test, targets_test, num_classes)


    print(
        '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f, MCE = %.4f, Brier = %.5f, AUROC = %.4f' %
        (ac, ECE, MCE, brier, AUROC)
    )     

    return top1.avg, top5.avg


       

        print(f"acc1s are {top1.avg.item()}")
        print(f"acc5s are {top5.avg.item()}")

   
