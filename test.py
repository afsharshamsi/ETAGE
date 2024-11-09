import pickle
import os
import open_clip
import copy
import random
import ast
import torch
import json
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
import deeplake
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from decouple import config
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import CIFAR10
from model import evaluate_model, train_model_camelyon, train_model_cifar, evaluate_model_freeze, evaluate_model_cam_ensemble_freeze, averaging_model, evaluate_model_ensemble_uncertainty, LN_true
from utilities import Paths, calculate_metrics, brier_score, calculate_auroc_multiclass
from preprocessor import load_data_camelyon, load_data_cifar, load_data_places#, CorruptedCIFARDataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from bayes_wrap import validate,  generate_freezed_particles, adapt_BTTA



from methods import tent, eata, sam, sar, deyo
from utils.utils import get_logger
from utils.cli_utils import *
import time
import wandb

import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

seed = 2295
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


''' -----------------------   Set path ------------------------------'''
paths = Paths(config)
paths.create_path()


''' -----------------------   loading CLIP ViT ------------------------------'''
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# mdl, preprocess = clip.load('ViT-B/32', device)
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
opt= 1



if config('dataset_name').upper() == "CIFAR10":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/tower2/DATA4/Afshar/datasets/cifar10/" + "cifar-10-batches-py")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False, transform=preprocess)

    corrupted_testset = np.load(f"Data/{config('corruption')}.npy")
    lbls = np.load("Data/labels.npy")
    test.data = corrupted_testset
    test.targets = lbls
    test.transform = preprocess

    # print(f'len test: {len(test)}')
    print('cifar10 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

elif config('dataset_name').upper() == "CIFAR100":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("/media/tower2/DATA4/Afshar/datasets/cifar100/" + "cifar-100-batches-py")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)

    corrupted_testset = np.load(f"Data/cifar100-c/{config('corruption')}.npy")
    lbls = np.load("Data/cifar100-c/labels.npy")
    test.data = corrupted_testset
    test.targets = lbls
    test.transform = preprocess

    print('cifar100 loaded')
    trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)


elif config('dataset_name').upper() == "DOMAINNET":

    ''' -----------------------   Loading the Data   ----------------------- '''
    train_data = deeplake.load("hub://activeloop/domainnet-clip-train")

    test_data = deeplake.load("hub://activeloop/domainnet-clip-test")

    print('Domainnet has been loaded')
    print(f'len train is {len(train_data)}')
    print(f'len test is {len(test_data)}')

    trainloaders, validation_loader, test_loader = load_data_places(preprocess, train_data, test_data, test_data, device)


############################################################################################
##################################################################################################################

if config('dataset_name').upper() == "CIFAR10":
    path_address = 'Model/max_min_cifar10/'

elif config('dataset_name').upper() == "CIFAR100":
    path_address = 'Model/max_min_cifar100/'


mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
ens_addr = [f for f in os.listdir(path_address) if f[-4:]=='1.pt']



def load_ensemble(ens_addr):
    ensemble=[]
    for i in range(opt):
        for i, addrr in enumerate(ens_addr):
            mdl, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            # mdl_addr = f'mdl-cam3/best_model_{i}_noise_std_0_series_0.pt'

            classification_head = get_classification_head()
            image_encoder = ImageEncoder(mdl)#, keep_lang=False)
            NET = ImageClassifier(image_encoder, classification_head)
            NET.freeze_head()

            model_new = copy.deepcopy(NET)
            fine_tuned_weights = torch.load(path_address + addrr)

            model_new.load_state_dict(fine_tuned_weights)
            ensemble.append(model_new)

    # print(f'number of models loaded is {len(ensemble)}')
    return ensemble

if  config('method').lower()!='btta':
    ensemble = load_ensemble(ens_addr)

e_margin = float(config('e_margin'))
sar_margin_e0 = float(config('sar_margin_e0'))
deyo_margin = float(config('deyo_margin'))
deyo_margin_e0 = float(config('deyo_margin_e0'))

e_margin *= math.log(float(config('num_class')))
sar_margin_e0 *= math.log(float(config('num_class')))
deyo_margin *= math.log(float(config('num_class'))) # for thresholding
deyo_margin_e0 *= math.log(float(config('num_class'))) # for reweighting tuning

if config('method').lower()=='tent':
    print(f"method: {config('method')}, corruption: {config('corruption')}")
    net = tent.configure_model(ensemble[0].to(device))
    params, param_names = tent.collect_params(net)
    # print(param_names)

    optimizer = torch.optim.SGD(params, float(config('learning_rate')), momentum=0.9) 
    tented_model = tent.Tent(net, optimizer)

    acc1, acc5 = validate(test_loader, tented_model, device, mode='eval')


elif config('method').lower() == "no_adapt":
    print(f"method: {config('method')}, corruption: {config('corruption')}")
    tented_model = ensemble[0].to(device)
    acc1, acc5 = validate(test_loader, tented_model, device, mode='eval')


elif config('method').lower() == "eata":
    print(f"method: {config('method')}, corruption: {config('corruption')}")
    if config("eata_fishers"):
        print('EATA!')

        net = eata.configure_model(ensemble[0].to(device))
        params, param_names = eata.collect_params(net)
        # print(param_names)

        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
        st_time = time.time()
        for iter_, data in enumerate(test_loader, start=1):

            images, targets = data[0], data[1]
            images, targets = images.to(device), targets.to(device)

            outputs = net(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(test_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
            print(f'\r fisher iter {iter_}/{len(test_loader)}', end='')
        print("\ncompute fisher matrices finished")
        del ewc_optimizer
    else:
        net = eata.configure_model(ensemble[0].to(device))
        params, param_names = eata.collect_params(net)
        print('ETA!')
        fishers = None
    
    end_time = time.time()

    print(f'time1: {end_time - st_time}')
    optimizer = torch.optim.SGD(params, float(config('learning_rate')), momentum=0.9)
    adapt_model = eata.EATA( net, optimizer, fishers, int(config('fisher_alpha')), e_margin = e_margin, d_margin=float(config('d_margin')))
    acc1, acc5 = validate(test_loader, adapt_model, device, mode='eval')


elif config('method').lower() == "sar":

    print(f"method: {config('method')}, corruption: {config('corruption')}")
    biased = False
    wandb_log = False

    st_time = time.time()

    net = sar.configure_model(ensemble[0].to(device))
    params, param_names = sar.collect_params(net)
    # print(param_names)

    base_optimizer = torch.optim.SGD
    optimizer = sam.SAM(params, base_optimizer, lr=float(config('learning_rate')), momentum=0.9)
    adapt_model = sar.SAR(net, optimizer, margin_e0= sar_margin_e0)

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if biased:
        LL_AM = AverageMeter('LL Acc', ':6.2f')
        LS_AM = AverageMeter('LS Acc', ':6.2f')
        SL_AM = AverageMeter('SL Acc', ':6.2f')
        SS_AM = AverageMeter('SS Acc', ':6.2f')
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
    
    end = time.time()
    correct_count = [0,0,0,0]
    total_count = [1e-6,1e-6,1e-6,1e-6]
    logits_test, targets_test= [], []
    for i, (images, target) in enumerate(test_loader):
        # images, target = dl[0], dl[1]
        images, target = images.to(device), target.to(device)

        if biased:
            if config('dataset_name')=='Waterbirds':
                place = dl[2]['place'].cuda()
            else:
                place = dl[2].cuda()
            group = 2*target + place
        output = adapt_model(images)
        if biased:
            TFtensor = (output.argmax(dim=1)==target)
            
            for group_idx in range(4):
                correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                total_count[group_idx] += len(TFtensor[group==group_idx])
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        logits_test.append(output.cpu().detach().numpy())
        targets_test.append(target.cpu().detach().numpy())           

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
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
                    wandb.log({f'{config("corruption")}/LL': LL,
                                f'{config("corruption")}/LS': LS,
                                f'{config("corruption")}/SL': SL,
                                f'{config("corruption")}/SS': SS,
                                })
            if wandb_log:
                wandb.log({f'{config("corruption")}/top1': top1.avg,
                            f'{config("corruption")}/top5': top5.avg
                            })

        if (i+1) % float(config('wandb_interval')) == 0:
            progress.display(i)

    acc1 = top1.avg
    acc5 = top5.avg
    
    if biased:
        print(f"- Detailed result under {config('corruption')}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
        if wandb_log:
            wandb.log({'final_avg/LL': LL,
                        'final_avg/LS': LS,
                        'final_avg/SL': SL,
                        'final_avg/SS': SS,
                        'final_avg/AVG': (LL+LS+SL+SS)/4,
                        'final_avg/WORST': min(LL,LS,SL,SS),
                        })

        avg = (LL+LS+SL+SS)/4
        print(f"Result under {config('corruption')}. The adaptation accuracy of SAR is  average: {avg:.5f}")

    else:

        en_time = time.time()
        print(f'time: {en_time - st_time}')
        print(f"acc1s are {top1.avg.item()}")
        print(f"acc5s are {top5.avg.item()}")

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

elif config('method').lower() == "deyo":

    print(f"method: {config('method')}, aug_type: {config('aug_type')}, corruption: {config('corruption')}")
    biased = False
    wandb_log = False       

    st_time = time.time()

    net = deyo.configure_model(ensemble[0].to(device))
    params, param_names = deyo.collect_params(net)
    # print(param_names)

    optimizer = torch.optim.SGD(params, float(config('learning_rate')), momentum=0.9)
    adapt_model = deyo.DeYO(net, optimizer, deyo_margin= deyo_margin, margin_e0= deyo_margin_e0)

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if biased:
        LL_AM = AverageMeter('LL Acc', ':6.2f')
        LS_AM = AverageMeter('LS Acc', ':6.2f')
        SL_AM = AverageMeter('SL Acc', ':6.2f')
        SS_AM = AverageMeter('SS Acc', ':6.2f')
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
    end = time.time()
    count_backward = 1e-6
    final_count_backward =1e-6
    count_corr_pl_1 = 0
    count_corr_pl_2 = 0
    total_count_backward = 1e-6
    total_final_count_backward =1e-6
    total_count_corr_pl_1 = 0
    total_count_corr_pl_2 = 0
    correct_count = [0,0,0,0]
    total_count = [1e-6,1e-6,1e-6,1e-6]
    logits_test, targets_test= [], []
    for i, dl in enumerate(test_loader):
        images, target = dl[0], dl[1]
        images, target = images.to(device), target.to(device)

        if biased:
            if config('dataset_name')=='Waterbirds':
                place = dl[2]['place'].cuda()
            else:
                place = dl[2].cuda()
            group = 2*target + place
        else:
            group=None

        output, backward, final_backward, corr_pl_1, corr_pl_2 = adapt_model(images, i, target, group=group)
        if biased:
            TFtensor = (output.argmax(dim=1)==target)
            
            for group_idx in range(4):
                correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                total_count[group_idx] += len(TFtensor[group==group_idx])
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        logits_test.append(output.cpu().detach().numpy())
        targets_test.append(target.cpu().detach().numpy())  
             
        count_backward += backward
        final_count_backward += final_backward
        total_count_backward += backward
        total_final_count_backward += final_backward
        
        count_corr_pl_1 += corr_pl_1
        count_corr_pl_2 += corr_pl_2
        total_count_corr_pl_1 += corr_pl_1
        total_count_corr_pl_2 += corr_pl_2

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
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
                wandb.log({f'{config("corruption")}/top1': top1.avg,
                            f'{config("corruption")}/top5': top5.avg,
                            f'acc_pl_1': count_corr_pl_1/count_backward,
                            f'acc_pl_2': count_corr_pl_2/final_count_backward,
                            f'count_backward': count_backward,
                            f'final_count_backward': final_count_backward})
            
            count_backward = 1e-6
            final_count_backward =1e-6
            count_corr_pl_1 = 0
            count_corr_pl_2 = 0

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % int(config('wandb_interval')) == 0:
            progress.display(i)

    acc1 = top1.avg
    acc5 = top5.avg
    
    if biased:
        print(f"- Detailed result under {config('corruption')}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
        if wandb_log:
            wandb.log({'final_avg/LL': LL,
                        'final_avg/LS': LS,
                        'final_avg/SL': SL,
                        'final_avg/SS': SS,
                        'final_avg/AVG': (LL+LS+SL+SS)/4,
                        'final_avg/WORST': min(LL,LS,SL,SS),
                        })
        
    if wandb_log:
        wandb.log({f'{config("corruption")}/top1': acc1,
                    f'{config("corruption")}/top5': acc5,
                    f'total_acc_pl_1': total_count_corr_pl_1/total_count_backward,
                    f'total_acc_pl_2': total_count_corr_pl_2/total_final_count_backward,
                    f'total_count_backward': total_count_backward,
                    f'total_final_count_backward': total_final_count_backward})

    if biased:
        avg = (LL+LS+SL+SS)/4
        print(f"Result under {config('corruption')}. The adaptation accuracy of DeYO is  average: {avg:.5f}")

        # LLs.append(LL)
        # LSs.append(LS)
        # SLs.append(SL)
        # SSs.append(SS)
        # acc1s.append(avg)
        # acc5s.append(min(LL,LS,SL,SS))

        # print(f"The LL accuracy are {LLs}")
        # print(f"The LS accuracy are {LSs}")
        # print(f"The SL accuracy are {SLs}")
        # print(f"The SS accuracy are {SSs}")
        # print(f"The average accuracy are {acc1s}")
        # print(f"The worst accuracy are {acc5s}")
    else:
        en_time = time.time()

        print(f'time: {en_time - st_time}')
        print(f"acc1s are {top1.avg.item()}")
        print(f"acc5s are {top5.avg.item()}")

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

elif config('method').lower() == "etage":

    print(f"method: {config('method')}, corruption: {config('corruption')}")
    ensemble = load_ensemble(ens_addr)

    for model in ensemble:
        LN_true(model)

    adapt_BTTA(ensemble, test_loader, device, config)
         
