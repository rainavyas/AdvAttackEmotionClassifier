'''
1) Perform PGD attack in first embedding layer
2) Visualize encoder embedding space in PCA decomposition
3) Analyze error size
4) F1 score using residue linear detector
5) F1 score using uncertainty
6) F1 score using Mahalanobis Distance
'''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve
import sys
import os
import argparse
from models import ElectraSequenceClassifier
from layer_handler import Electra_Layer_Handler
from data_prep_sentences import get_test
from transformers import ElectraTokenizer
from tools import accuracy_topk, AverageMeter, get_default_device
import matplotlib.pyplot as plt
from average_comp_dist import stds, get_all_comps, get_avg_comps
import numpy as np
from linear_pca_classifier import LayerClassifier
from precision_recall_linear_classifier import get_best_f_score
from uncertainty import ensemble_uncertainties_classification


class Attack(torch.nn.Module):
  def __init__(self, attack_init):
    super(Attack, self).__init__()
    self.attack = torch.nn.Parameter(attack_init, requires_grad=True)

  def forward(self, X, attention_mask, layer_handler, device=torch.device('cpu')):
    X_attacked = X + self.attack
    y = layer_handler.pass_through_rest(X_attacked, attention_mask, device)
    return y

def clip_params(model, epsilon):
    old_params = {}

    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    old_params['attack'][old_params['attack']>epsilon] = epsilon
    old_params['attack'][old_params['attack']<(-1*epsilon)] = -1*epsilon
 
    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])

def train_pgd(dl, attack_model, criterion, optimizer, epoch, epsilon, layer_handler, device):
    """
        Run one train epoch
    """
    
    losses = AverageMeter()
    top1 = AverageMeter()

    for input, mask, target in dl:

        attack_model.train()
        X = input.to(device)
        attention_mask = mask.to(device)
        target = target.to(device)

        # compute output
        output = attack_model(X, attention_mask, layer_handler, device)
        loss = criterion(output, target)
        loss_neg = -1*loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_neg.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            # clip the parameters
            clip_params(attack_model, epsilon)

            output = output.float()
            loss = loss_neg.float()

            # measure accuracy
            prec1 = accuracy_topk(output.data, target)

            losses.update(loss.item(), X.size(0))
            top1.update(prec1.item(), X.size(0))

    print(f'Epoch: {epoch}\t Loss: {losses.avg}\t Accuracy: {top1.avg}')


def eval_pgd(X, attention_mask, target, attack_model, criterion, layer_handler):
    attack_model.eval()
    with torch.no_grad():
        output_attack = attack_model(X, attention_mask, layer_handler)
        loss = criterion(output_attack, target)
        output_attack = output_attack.float()
        output_no_attack = layer_handler.pass_through_rest(X, attention_mask)
        fool = fooling_rate(output_no_attack, output_attack, target)
        prec1 = accuracy_topk(output_attack.data, target)
    print(f'Evaluation\t Loss: {loss}\t Accuracy: {prec1}\t Fooling Rate: {fool}')

def fooling_rate(output_no_attack, output_attack, target):

    original_pred = torch.argmax(output_no_attack, dim=1)
    attack_pred = torch.argmax(output_attack, dim=1)

    total_count = 0
    fool_count = 0
    for orig, att, targ in zip(original_pred, attack_pred, target):
        if orig != targ:
            continue
    
        if att != orig:
            fool_count+=1
        total_count+=1
    return fool_count/total_count

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=1):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            text = '\n Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, prec=accs)
            print(text)

def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (x, target) in enumerate(val_loader):

            x = x.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

    text ='\n Test\t Loss ({loss.avg:.4f})\t Accuracy ({prec.avg:.3f})\n'.format(loss=losses, prec=accs)
    print(text)

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DATA_PATH', type=str, help='data filepath')
    commandLineParser.add_argument('EIGENVECTORS', type=str, help='Learnt eigenvectors .pt file for PCA projection')
    commandLineParser.add_argument('CORRECTION_MEAN', type=str, help='Learnt correction mean.pt file for PCA projection')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('ENSEMBLE_DIR', type=str, help='trained sentiment classifier .th models dir for ensemble uncertainty')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="tweet index to start at")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help="tweet index to end at")
    commandLineParser.add_argument('--epsilon', type=float, default=0.01, help="l-inf pgd perturbation size")
    commandLineParser.add_argument('--lr', type=float, default=0.1, help="pgd learning rate")
    commandLineParser.add_argument('--epochs', type=int, default=20, help="Number of epochs for PGD attacks")
    commandLineParser.add_argument('--seed', type=int, default=1, help="seed for randomness")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    commandLineParser.add_argument('--layer_num', type=int, default=12, help="Layer at which to use detector")


    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_path = args.DATA_PATH
    eigenvectors_path = args.EIGENVECTORS
    correction_mean_path = args.CORRECTION_MEAN
    out_file = args.OUT_FILE
    ensemble_dir = args.ENSEMBLE_DIR
    start_ind = args.start_ind
    end_ind = args.end_ind
    epsilon = args.epsilon
    lr = args.lr
    epochs = args.epochs
    seed = args.seed
    cpu_use = args.cpu
    layer_num = args.layer_num

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pgd_attack_linear_detector.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Set seed
    torch.manual_seed(seed)

    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    # Get all data
    tweets_list, labels = get_test('electra', data_path)
    tweets_list = tweets_list[start_ind:end_ind]
    labels = labels[start_ind:end_ind]

    # -----------------------------------
    # 1) Individual PGD attack
    # -----------------------------------

    # Create model handler
    handler = Electra_Layer_Handler(model, layer_num=0)

    # Prepare data as tensors
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    encoded_inputs = tokenizer(tweets_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    labels = torch.LongTensor(labels)

    # Map ids to input layer embeddings
    input_embeddings = handler.get_layern_outputs(ids, mask)
    print(torch.mean(torch.abs(input_embeddings)))

    # Create attack model
    attack_init = torch.zeros_like(input_embeddings)
    attack_model = Attack(attack_init)
    attack_model.to(device)

    # use dl
    ds = TensorDataset(input_embeddings, mask, labels)
    dl = DataLoader(ds, batch_size=input_embeddings.size(0), shuffle=False)

    # Perform PGD attack
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_pgd(dl, attack_model, criterion, optimizer, epoch, epsilon, handler, device)
    eval_pgd(input_embeddings, mask, labels, attack_model, criterion, handler)


    # ---------------------
    # 2) Visualize PCA embedding space decomposition
    # ---------------------

    # Load the eigenvectors for PCA decomposition and the correction mean
    eigenvectors = torch.load(eigenvectors_path)
    correction_mean = torch.load(correction_mean_path)

    # Get embeddings
    original_embeddings = handler.pass_through_some(input_embeddings, mask, output_layer=layer_num)[:,0,:].squeeze(dim=1)
    attack_embeddings = handler.pass_through_some(input_embeddings+attack_model.attack, mask, output_layer=layer_num)[:,0,:].squeeze(dim=1)

    # Get average components against rank
    original_avg_comps = get_avg_comps(original_embeddings, eigenvectors, correction_mean)
    attack_avg_comps = get_avg_comps(attack_embeddings, eigenvectors, correction_mean)

    # Plot the results
    ranks = np.arange(len(original_avg_comps))
    plt.plot(ranks, original_avg_comps, label='Original')
    plt.plot(ranks, attack_avg_comps, label='Attacked')
    plt.yscale('log')
    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Average Component Size')
    plt.legend()
    plt.savefig(out_file)
    plt.clf()

    # Report std diff between attack and original curves
    original_comps = get_all_comps(original_embeddings, eigenvectors, correction_mean)
    attack_comps = get_all_comps(attack_embeddings, eigenvectors, correction_mean)

    std_diffs = stds(original_comps, attack_comps)
    print("OOD metric", torch.mean(std_diffs))

    # Plot std_diffs ranked by size
    std_diffs_ordered, _ = torch.sort(std_diffs)
    ranks = np.arange(len(std_diffs_ordered))
    plt.plot(ranks, std_diffs_ordered)
    plt.xlabel('std difference rank')
    plt.ylabel('std difference')
    plt.savefig('std_'+out_file)
    plt.clf()

    # --------------------------------------
    # 3) Analyse error size
    # --------------------------------------

    # Report the following average errors:
        # - l2
        # - l-inf

    print()
    diffs = torch.reshape(torch.abs(attack_model.attack * torch.unsqueeze(mask, dim=-1).expand(-1,-1,attack_model.attack.size(-1))), (attack_model.attack.size(0), -1))

    l2s = torch.sqrt(torch.sum(diffs**2, dim=1))
    print(f'l2: mean={torch.mean(l2s)} std={torch.std(l2s)}')

    linfs, _ = torch.max(diffs, dim=1)
    print(f'l-inf: mean={torch.mean(linfs)} std={torch.std(linfs)}')

    # --------------------------------------
    # 3) F1 score using Residue Linear Detector
    # --------------------------------------

    train_ratio = 0.6
    bs = 16
    epochs = 20

    train_original_comps = original_comps[:int(train_ratio*len(original_comps))]
    test_original_comps = original_comps[int(train_ratio*len(original_comps)):]
    train_attack_comps = attack_comps[:int(train_ratio*len(attack_comps))]
    test_attack_comps = attack_comps[int(train_ratio*len(attack_comps)):]

    X_train = torch.FloatTensor(np.concatenate((train_original_comps, train_attack_comps), axis=0))
    y_train = torch.LongTensor([0]*len(train_original_comps)+[1]*len(train_attack_comps))
    ds_train = TensorDataset(X_train, y_train)
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)

    X_test = torch.FloatTensor(np.concatenate((test_original_comps, test_attack_comps), axis=0))
    y_test = torch.LongTensor([0]*len(test_original_comps)+[1]*len(test_attack_comps))
    ds_test = TensorDataset(X_test, y_test)
    dl_test = DataLoader(ds_test, batch_size=bs)

    # Model
    detector = LayerClassifier(X_train.size(-1))
    detector.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    print('--------')
    print('Residue Linear Classifier Training')
    print('---------')

    # Train
    for epoch in range(epochs):

        # train for one epoch
        text = '\n current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        with open(out_file, 'a') as f:
            f.write(text)
        print(text)
        train(dl_train, detector, criterion, optimizer, epoch, device)

        # evaluate
        eval(dl_test, detector, criterion, device)
    
    # Get F1 score using linear detector prediction probs
    detector.to(torch.device('cpu'))
    with torch.no_grad():
        logits = detector(X_test)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        adv_probs = probs[:,1].squeeze().cpu().detach().numpy()
    
    precision, recall, _ = precision_recall_curve(y_test.cpu().detach().numpy(), adv_probs)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print(f'F1 score Residue Linear Detector: {best_f1}')
    

    # --------------------------------------
    # 4) F1 score using Uncertainty
    # --------------------------------------
    print('--------')
    print('Uncertainty Detection Approach')
    print('---------')

    # Load Ensemble of Models and evaluate each
    ensemble_size = 5
    basename = 'electra_trained_seed'
    logits_by_model_original = []
    logits_by_model_attack = []



    # --------------------------------------
    # 5) F1 score using Mahalanobis Distance
    # --------------------------------------

    # Unnecessary to do here - experiment in other script gives F1 = 0.68






