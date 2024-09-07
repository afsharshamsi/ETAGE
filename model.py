import torch
import math
import open_clip
import copy
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from decouple import config
# from utils import add_noise_to_parameters, cosine_lr
import numpy as np
import os
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
# from bayes_wrap import BayesWrap
import torch.optim.lr_scheduler as lr_scheduler
# from utils import generate_results


def train_model_camelyon(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()


    opt = config('opt')
    model = BayesWrap(NET, opt)

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=float(config('learning_rate')))


    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, step = 0., 0.
        for i, (img, labels, metadata) in enumerate(train_loader):
            
            img, labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            kwargs = {"return_entropy": False}
            logits, soft_out = model(img, **kwargs)

            loss = criterion(logits, labels)
            loss.backward()
            model.update_grads()

            losses += loss.item()
            step += 1
            
            optimizer.step()     
            optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        

        loss_epoch = losses / step
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, step2 = 0., 0.
        for img, text, metadata in validation_loader:
            img, text = img.cuda(), text.cuda()
            logits, soft_out = model(img, **kwargs)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(soft_out, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            step2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / step2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        # Save checkpoint if the current validation loss is the best
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
            for i, particle in enumerate(model.particles):
                torch.save(particle.state_dict(), f'Model/best_model_{i}_noise_std_{noise_std}.pt')

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels, metadata in test_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out, stds = model(img, **kwargs)

            predicted = torch.argmax(soft_out, dim=1)
            all_scores.extend(predicted.cpu().numpy()) 
            all_softs.extend(soft_out.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    # all_scores_train = []
    # all_labels_train = []
    # all_entropies_train=[]

    # i = 0
    # with torch.no_grad():

    #     for images, labels, metadata in train_loader:
    #         img , text = images.cuda(), labels.cuda()
    #         kwargs = {"return_entropy": True}
    #         logits, entropies, soft_out,_ = model(img, **kwargs)

    #         predicted = torch.argmax(soft_out, dim=1)
    #         all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
    #         all_labels_train.extend(labels.numpy())  # Extend the list with true labels
    #         all_entropies_train.extend(entropies.cpu().numpy())
    #         print(f'\r calculating entropies for train {i}', end='')
    #         i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()
        # all_scores_train = np.array(all_scores_train).tolist()
        # all_labels_train = np.array(all_labels_train).tolist()
        # all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test": all_softs,
                     "all_stds_test": all_stds
                                                }

    labels_info_path = f"Results/entropies_model_{j}_noise_std_{noise_std}_cam.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    # Save best model
    # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
    # torch.save(best_model, best_model_path)

    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_cam_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels





def train_model_cifar(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()


    opt = config('opt')
    model = BayesWrap(NET, opt)

    model = model.cuda()

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=float(config('learning_rate')))

    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, step = 0., 0.
        for i, (img, labels) in enumerate(train_loader):
            
            img, labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            kwargs = {"return_entropy": False}
            logits, soft_out = model(img, **kwargs)

            loss = criterion(logits, labels)
            loss.backward()
            model.update_grads()

            losses += loss.item()
            step += 1
            
            optimizer.step()     
            # optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / step
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, step2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.cuda(), text.cuda()
            logits, soft_out = model(img, **kwargs)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            step2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / step2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out, stds = model(img, **kwargs)

            predicted = torch.argmax(soft_out, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(soft_out.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores_train = []
    all_labels_train = []
    all_entropies_train=[]

    i = 0
    with torch.no_grad():

        for images, labels in train_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out, _ = model(img, **kwargs)

            predicted = torch.argmax(soft_out, dim=1)
            all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels_train.extend(labels.numpy())  # Extend the list with true labels
            all_entropies_train.extend(entropies.cpu().numpy())
            print(f'\r calculating entropies for train {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()
        all_scores_train = np.array(all_scores_train).tolist()
        all_labels_train = np.array(all_labels_train).tolist()
        all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds,
                     "all_labels_train": all_labels_train,
                     "all_scores_train": all_scores_train,
                     "all_entropies_train": all_entropies_train}

    labels_info_path = f"Results/model_{j}_noise_std_{noise_std}_cifar_entropies.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_cifar_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels




def evaluate_model(model, test_loader, text_inputs, device):

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            # img, text = img.cuda(), text.cuda()
            img_feats = model.encode_image(img)
            text_feats = model.encode_text(text_inputs)

            # img_feats /= img_feats.norm(dim=-1, keepdim=True)
            # text_feats /= text_feats.norm(dim=-1, keepdim=True)
            logits = torch.matmul(img_feats, text_feats.T)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    return  all_scores, all_labels    

def evaluate_model_freeze(model, test_loader, device):
    

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies = []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            model = model.to(device)
            img , text = images.to(device), labels.to(device)
            logits = model(img)

            sft = torch.softmax(logits, 1)

            entropies = (-sft * torch.log(sft + 1e-8)).sum(1)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            print(f'\r {i}', end='')
            i +=1

        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_entropies = np.array(all_entropies)
    return  all_scores, all_labels, all_entropies.mean(0)   


def evaluate_model_cam_ensemble_freeze(ensemble, test_loader, device):

    # model.eval()  # Set the model to evaluation mode


    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    i = 0
    with torch.no_grad():
      
        for images, labels in test_loader:
            
            img , text = images.to(device), labels.to(device)
 
            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:
 
                model = model.to(device)
                l = model(img)
                sft = torch.softmax(l, 1)
                # sft= sft*coef
                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
                
            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            entropies = torch.stack(entropies).mean(0)

            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        # all_scores = np.array(all_scores)
        # all_labels = np.array(all_labels)

    #     all_scores = np.array(all_scores).tolist()
    #     all_labels = np.array(all_labels).tolist()
    #     all_entropies = np.array(all_entropies).tolist()
    #     all_softs = np.array(all_softs).tolist()
    #     all_stds = np.array(all_stds).tolist()


    # labels_info = {  "all_labels_test": all_labels,
    #                  "all_scores_test": all_scores,   
    #                  "all_entropies_test":  all_entropies,
    #                  "all_softs_test":  all_softs,
    #                  "all_std_test": all_stds
    #                                             }

    # labels_info_path = f"Results/entropies_{config('dataset_name')}.json"
    # with open(labels_info_path, 'w') as fp:
    #     json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 


#-----------------------------------------------------------------------------------------------------------------


def evaluate_cam_mix_ensemble_soup(ensemble, soup_model, a, test_loader, device):

 
    soup_model.eval()
    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_entropies_soup = [], [], [], []
    coefficients=[0.25, 0.25, 0.25, 0.25, 0.25]
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            
            logits, entropies, soft_out, entropies_soup = [], [], [], []
            for coef, model in zip(coefficients, ensemble):
                model.eval()
                model.cuda()
                l=model(img)

                # logits.append(sft)
                sft = torch.softmax(l, 1)
                sft = sft*coef
                # entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                soft_out.append(sft)
                logits.append(l)


            # Averaging the softmax outputs for ensemble models
            ensemble_avg_soft_out = torch.stack(soft_out).mean(0)
            weighted_ensemble_avg_soft_out = a * ensemble_avg_soft_out

            # # Calculating softmax output for soup model
            soup_model.cuda()
            l_soup = soup_model(img)
            sft_soup = torch.softmax(l_soup, 1)
            # entropies_soup.append((-sft_soup * torch.log(sft_soup + 1e-8)).sum(1))
            
            # Weighting the softmax output for the soup model
            weighted_soup_soft_out = (1 - a) * sft_soup

            final_soft_out =   weighted_soup_soft_out + weighted_ensemble_avg_soft_out


            ent1 =  (-final_soft_out * torch.log(final_soft_out + 1e-8)).sum(1)
            # ent = ent1.mean(0)
            # entropies = torch.stack(entropies).mean(0)
            
            predicted = torch.argmax(final_soft_out, dim=1)
            # all_entropies_soup.extend(entropies_soup.cpu().numpy())
            all_scores.extend(predicted.cpu().numpy())  
            all_entropies.extend(ent1.cpu().numpy())
            all_softs.extend(final_soft_out.cpu().numpy())
            all_labels.extend(labels.numpy())  
            # all_stds.extend(stds.cpu().numpy())
            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_entropies = np.array(all_entropies)
        print(f'entropy for a:{a} is {all_entropies.mean(0)}')

    return  all_scores, all_labels , all_entropies.mean(0)

#-----------------------------------------------------------------------------------------------------------------

def print_param_info(model, model_name):
    print(f"Info for {model_name}:")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(f"Parameter: {name}, Total Elements: {param.numel()}, Gradient: {'Yes' if param.grad is not None else 'No'}")
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")



def soup_diff(soup_model, ensemble, test_loader, device):
    soup_model.eval()
    soup_model.to(device)


    for model in ensemble:
        model.train()
        model.to(device)

    total_loss = 0.0
    total_correct_adjusted = 0
    total_correct_individual_mix = 0
    total_samples = 0
    i = 0

    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    all_entropies_individual, all_entropies_soup = [], []
    for images, labels, _ in test_loader:
        imgs = images.to(device)
        lbls = labels.to(device)

        # imgs.requires_grad_(True)

        # with torch.no_grad():  
        soup_output = soup_model(imgs)
        gradients_soup = torch.autograd.grad(outputs=soup_output, inputs=[param for param in soup_model.parameters() if param.requires_grad], grad_outputs=torch.ones_like(soup_output), allow_unused=True)
        gradient_vector_soup = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(gradients_soup, soup_model.parameters())])

        # soup_output = torch.softmax(soup_output, 1)
        # ent_soup =  (-soup_output * torch.log(soup_output + 1e-8)).sum(1)
        # # print(f'entropy of soup is {ent_soup}')
        # all_entropies_soup.extend(ent_soup.cpu().detach().numpy())     
        logits = []
        predicted_outputs = []
        for individual_model in ensemble:
            individual_output = individual_model(imgs)
            logits.append(individual_output)
            # gradients_individual = torch.autograd.grad(outputs=individual_output, inputs=[param for param in individual_model.parameters() if param.requires_grad], grad_outputs=torch.ones_like(individual_output), allow_unused=True)
            # gradient_vector = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(gradients_individual, individual_model.parameters())])

            param_diff = []
            for (param_soup, param_individual) in zip(soup_model.parameters(), individual_model.parameters()):
                if param_soup.requires_grad and param_individual.requires_grad:
                    param_diff.append((param_soup - param_individual).view(-1))
            param_diff_vector = torch.cat(param_diff)

            adjusted_output = soup_output + torch.dot(gradient_vector_soup, param_diff_vector)
            predicted_outputs.append(adjusted_output)

        logits = torch.stack(logits).mean(0)
        predicted_output = torch.stack(predicted_outputs).mean(0)
        predicted_output = torch.softmax(predicted_output, 1)

        # ent_individual =  (-predicted_output * torch.log(predicted_output + 1e-8)).sum(1)
        
        # print(f'entropy of individual is {ent_individual}')
        # all_entropies_individual.extend(ent_individual.cpu().detach().numpy())
        # ent = ent1.mean(0)
        # Calculate Cross-Entropy Loss using pseudo labels
        # pseudo_labels = torch.argmax(soup_output, dim=1)
        # loss = cross_entropy_loss_fn(predicted_output, lbls)
        # total_loss += loss.item() * imgs.size(0)

        # Calculate accuracy using original labels
        _, predicted_adjusted_output = torch.max(predicted_output, 1)
        _, predicted_individual_output = torch.max(logits, 1)
        # _, predicted_individual_2 = torch.max(individual_output_2, 1)
        total_correct_adjusted += (predicted_adjusted_output == lbls).sum().item()
        total_correct_individual_mix += (predicted_individual_output == lbls).sum().item()
        # total_correct_individual_2 += (predicted_individual_2 == lbls).sum().item()

        total_samples += imgs.size(0)

        print(f'\rBatch {i} processed', end='')
        i += 1

    # average_cross_entropy_loss = total_loss / total_samples
    accuracy_adjusted = total_correct_adjusted / total_samples
    accuracy_individual_ens = total_correct_individual_mix / total_samples
    # accuracy_individual_2 = total_correct_individual_2 / total_samples
    # print(f'all entropies of individual {np.mean(all_entropies_individual)}')
    # print(f'\nAverage Cross-Entropy Loss over all batches: {average_cross_entropy_loss}')
    # print(f'\n all entropies of soup {np.mean(all_entropies_soup)}')
    # print(f'all entropies of individual {np.mean(all_entropies_individual)}')
    # print(f"difference of entropies are {np.mean(all_entropies_soup)-np.mean(all_entropies_individual)}")
    print(f'adjusted soup Model Accuracy: {accuracy_adjusted}')
    print(f'Ensemble Accuracy: {accuracy_individual_ens}')
    print(f'difference of adjusted and ens: {accuracy_individual_ens-accuracy_adjusted}')



#-----------------------------------------------------------------------------------------------------------------
def averaging_model(model_address):

    ensemble=[]
    for i in range(len(model_address)):
        mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        # mdl_addr = f'nmdl/mdl_{i}.pt'
        classification_head = get_classification_head()
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)
        net = ImageClassifier(image_encoder, classification_head)
        net.freeze_head()

    
        model_new = copy.deepcopy(net)
        fine_tuned_weights = torch.load("./nmdl/"+ model_address[i])
        model_new.load_state_dict(fine_tuned_weights)
        ensemble.append(model_new)
        print(f'model {i} is loaded from {model_address[i]}')


    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    classification_head = get_classification_head()
    image_encoder = ImageEncoder(model)#, keep_lang=False)
    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()


    average_model = copy.deepcopy(net)

    state_dicts = [mdel.state_dict() for mdel in ensemble]

    average_state_dict = {}
    num_models = len(ensemble)

    # coefficients = [0.2172, 0.2368, 0.1828, 0.1974, 0.1658]


    for key in ensemble[0].state_dict():
        average_state_dict[key] =sum(state_dict[key] for state_dict in state_dicts) / num_models
        # average_state_dict[key] = sum([coeff * state_dict[key] for coeff, state_dict in zip(coefficients, state_dicts)])#/ len(coefficients)

    average_model.load_state_dict(average_state_dict)

    print('The averaged model will be used for comparison')
    print("")   

    return average_model, ensemble


#-----------------------------------------------------------------------------------------------------------------


def custom_loss(single_model_output, ensemble_soup_output):
    return F.kl_div(F.log_softmax(single_model_output, dim=1), F.softmax(ensemble_soup_output, dim=1), reduction='batchmean')



def best_combination(ensemble, soup_model, single_model, a, test_loader, device):
    init_accuracy = 0
    single_model.cuda()
    single_model.train()

    soup_model.eval()
    # Evaluation loop
    optimizer = torch.optim.SGD([p for p in single_model.parameters() if p.requires_grad], lr=0.0001, weight_decay=float(config("Weight_decay")))
    
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    scheduler = cosine_lr(
                                optimizer,
                                0.001,
                                int(config("warmup_length")),
                                int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
                            )


    all_scores = []
    all_labels, all_entropies, all_softs, all_entropies_soup = [], [], [], []
    i = 0
    for epoch in range(10):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (images, labels) in enumerate(test_loader):

            step = (
                i // int(config('num_grad_accumulation'))
                + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
                                                )


            img, labels = images.to(device), labels.to(device)

            logits, soft_out = [], []
            for model in ensemble:
                model.eval()
                model.cuda()
                l = model(img)
                sft = torch.softmax(l, 1)
                soft_out.append(sft)
                logits.append(l)

            # Averaging the softmax outputs for ensemble models
            ensemble_avg_soft_out = torch.stack(soft_out).mean(0)
            weighted_ensemble_avg_soft_out = a * ensemble_avg_soft_out

            # Calculating softmax output for soup model
            soup_model.cuda()
            l_soup = soup_model(img)
            sft_soup = torch.softmax(l_soup, 1)
            weighted_soup_soft_out = (1 - a) * sft_soup

            final_soft_out = weighted_ensemble_avg_soft_out + weighted_soup_soft_out 
            pseudo_labels = torch.argmax(final_soft_out, dim=1)
            
            
            single_logits = single_model(img)


            loss = criterion(single_logits, pseudo_labels)
            # loss = criterion(single_soft_out, final_soft_out.detach())

            # difference = single_soft_out - final_soft_out.detach()
            # loss = torch.sqrt(torch.sum(difference ** 2))
            # loss = torch.sum(torch.abs(difference))


            # Calculate accuracy
            _, predicted_labels = torch.max(single_logits, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # scheduler(step)
            optimizer.step()

            total_loss += loss.item()
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {len(test_loader)}, loss: {loss.item():.4f}", end='')

        average_loss = total_loss / len(test_loader)  # Calculate the average loss for the epoch
        accuracy = correct_predictions / total_samples  # Calculate accuracy
        print(" ")
        print(f'Epoch {epoch}, Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

        if accuracy > init_accuracy:
            init_accuracy = accuracy
            best_model = copy.deepcopy(single_model.state_dict())
            best_model_path = f"Model/best_single_model.pt"

    torch.save(best_model, best_model_path)
    print(f'Best model has been saved')     


#---------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import copy

import torch
import torch.nn as nn
from functorch import make_functional_with_buffers, vmap
import torch.nn.functional as F




# class WeightedModelAveraging(nn.Module):
#     def __init__(self, models):
#         super(WeightedModelAveraging, self).__init__()
#         self.models = models
#         # Initialize coefficients equally
#         self.coefficients = nn.Parameter(torch.ones(len(models)) / len(models))

#         self.params_list = [torch.nn.utils.parameters_to_vector(model.parameters()) for model in models]

#     def forward(self, x):
#         avg_params = torch.zeros_like(self.params_list[0])
#         for coeff, params in zip(self.coefficients, self.params_list):
#             avg_params += coeff * params

#         f_model, org_params, buffers = make_functional_with_buffers(self.models[0])

#         reshaped_params = []
#         offset = 0
#         for param in org_params:
#             num_elements = param.numel()
#             reshaped_params.append(avg_params[offset:offset + num_elements].view(param.shape))
#             offset += num_elements

#         output = f_model(reshaped_params, buffers, x)
#         return output



class WeightedModelAveraging(nn.Module):
    def __init__(self, models):
        super(WeightedModelAveraging, self).__init__()
        self.models = models
        initial_coeff = 0.95
        self.coeff = nn.Parameter(torch.tensor(initial_coeff))

        self.params_list = [torch.nn.utils.parameters_to_vector(model.parameters()) for model in models]

    def forward(self, x):
        coeff = self.coeff
        coeff2 = 1 - coeff
        
        avg_params = torch.zeros_like(self.params_list[0])
        for coeff, params in zip([coeff, coeff2], self.params_list):
            avg_params += coeff * params

        f_model, org_params, buffers = make_functional_with_buffers(self.models[0])

        reshaped_params = []
        offset = 0
        for param in org_params:
            num_elements = param.numel()
            reshaped_params.append(avg_params[offset:offset + num_elements].view(param.shape))
            offset += num_elements

        output = f_model(reshaped_params, buffers, x)
        return output



def optimizing_beta(ensemble, test_loader, device):
    for model in ensemble:
        model.to(device)
        model.eval()


    total_alpha=[]
    for i, (inputs, labels) in enumerate(test_loader):    
        t_alpha = 0

        weighted_model_averaging = WeightedModelAveraging(ensemble).to(device)
        optimizer = torch.optim.SGD([weighted_model_averaging.coeff], lr=0.2)    
        inputs, labels = inputs.to(device), labels.to(device)
        for epoch in range(6):      

            optimizer.zero_grad()
            
            if epoch==0:
            #entropy for model 1 
                 with torch.no_grad():
                    l_mdl_0 = ensemble[0](inputs)
                    sft_mdl0 = torch.softmax(l_mdl_0, 1)
                    mdl_0_entropy = (-sft_mdl0 * torch.log(sft_mdl0 + 1e-8)).sum(1)

                    #entropy for model 1 
                    l_mdl_1 = ensemble[1](inputs)
                    sft_mdl1 = torch.softmax(l_mdl_1, 1)
                    mdl_1_entropy = (-sft_mdl1 * torch.log(sft_mdl1 + 1e-8)).sum(1)


            # entropy of averaged model
            outputs = weighted_model_averaging(inputs)
            soft_averaged = torch.softmax(outputs, 1)
            averaged_entropy = (-soft_averaged * torch.log(soft_averaged + 1e-8)).sum(1)
            
            alpha = weighted_model_averaging.coeff
            beta = 1- alpha


            loss = -(alpha * mdl_0_entropy + beta * mdl_1_entropy - averaged_entropy)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                weighted_model_averaging.coeff.clamp_(min=0.01, max=0.99)
                # weighted_model_averaging.coefficients /= weighted_model_averaging.coefficients.sum()

            print(f"\r[Epoch: {epoch}], from 6, loss: {loss.item():.4f}", end='')   
        # t_all = t_alpha.item()  
        alph = alpha.item()
        total_alpha.append(alph)  

        print(f'\n instance {i}, alpha: {alph}')


    alphas_info_path = f"Results/alphas_gradient_cifar10.json"
    with open(alphas_info_path, 'w') as fp:
        json.dump(total_alpha, fp, indent=2)










def evaluate_model_ensemble_uncertainty(ensemble, test_loader, device):
    """Evalate for normal model (w/o lora)
    to get uncertainty
    """

    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_H, all_E_entropies, all_softs, all_stds = [], [], [], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:

            img , text = images.to(device), labels.to(device)

            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:

                model = model.cuda()
                l = model(img)
                sft = torch.softmax(l, 1)

                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)

            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            # this is stack of particle entropy
            entropies = torch.stack(entropies)
            # this is expected entropies
            E_entropies = entropies.mean(0)
            # get the entropy of expected probabilities
            H = (-softs * torch.log(softs + 1e-8)).sum(1)


            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_H.extend(H.cpu().numpy())
            all_E_entropies.extend(E_entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

            print(f'\r {i}', end='')
            i +=1

        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_H = np.array(all_H).tolist()
        all_E_entropies = np.array(all_E_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_H_test":  all_H,
                     "all_E_entropies_test":  all_E_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds
                                                }

    labels_info_path = "Results/ent_MI_{}.json".format(config('dataset_name'))
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 

def LN_true(model):
    for name, param in model.named_parameters():
        
        if "ln" in name:
            param.requires_grad = True  
        else:
            param.requires_grad = False  
