'''
Use the CLS token at a specific layer as an
input into a simple linear classifier to distinguish between original and
adversarial samples
'''

import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tools import AverageMeter, get_default_device, accuracy_topk
from layer_handler import Electra_Layer_Handler
from models import ElectraSequenceClassifier
from transformers import ElectraTokenizer
import json
from data_prep import get_train

def batched_get_layer_embedding(sentences_list, handler, tokenizer, device, bs=8):
    '''
    Performs the function of get_layer_embedding from pca_bert_layer
    in batches and allows gpu use
    '''
    encoded_inputs = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    return batched_get_handler_embeddings(ids, mask, handler, device, bs=bs)


def batched_get_handler_embeddings(input_ids, mask, handler, device, bs=8):
    '''
    Input is a tensor of input ids and mask
    Returns tensor of CLS embeddings at the correct layer
    Does this in batches
    '''
    CLS = []
    ds = TensorDataset(input_ids, mask)
    dl = DataLoader(ds, batch_size=bs)
    with torch.no_grad():
        for id, m in dl:
            id = id.to(device)
            m = m.to(device)
            layer_embeddings = handler.get_layern_outputs(id, m, device=device)
            CLS_embeddings = layer_embeddings[:,0,:].squeeze(dim=1)
            CLS.append(CLS_embeddings.cpu())
    embeddings = torch.cat(CLS)
    return embeddings


def get_sentence(fname):
    failed = False
    try:
        with open(fname, 'r') as f:
            item = json.load(f)
    except:
        print("Failed to load", fname)
        failed = True
    if not failed:
        original_prob = item['original prob']
        updated_prob = item['updated prob']
        original_pred = original_prob.index(max(original_prob))
        updated_pred = updated_prob.index(max(updated_prob))
        label = int(item['true label'])
        if (original_pred == label) and (updated_pred != original_pred):
            original = item['sentence']
            attack = item['updated sentence']
        else:
            return None, None
    else:
        return None, None
    return original, attack

def load_test_adapted_data_sentences(base_dir, num_test):
    '''
    Excludes data points with incorrect original predictions
    '''
    original_list = []
    attack_list = []
    for i in range(num_test):
        fname = base_dir + '/'+str(i)+'.txt'
        original, attack = get_sentence(fname)
        if original is not None:
            original_list.append(original)
            attack_list.append(attack)

    return original_list, attack_list

def train(train_loader, model, criterion, optimizer, epoch, device, out_file, print_freq=1):
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
            with open(out_file, 'a') as f:
                f.write(text)

def eval(val_loader, model, criterion, device, out_file):
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
    with open(out_file, 'a') as f:
        f.write(text)

class LayerClassifier(nn.Module):
    '''
    Simple Linear classifier
    '''
    def __init__(self, dim, classes=2):
        super().__init__()
        self.layer = nn.Linear(dim, classes)
    def forward(self, X):
        return self.layer(X)


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('OUT', type=str, help='file to print results to')
    commandLineParser.add_argument('CLASSIFIER_OUT', type=str, help='.th to save linear adv attack classifier to')
    commandLineParser.add_argument('--layer_num', type=int, default=12, help="layer to investigate")
    commandLineParser.add_argument('--num_points_test', type=int, default=2000, help="number of data points to use test")
    commandLineParser.add_argument('--num_points_val', type=int, default=200, help="number of test data points to use for validation")
    commandLineParser.add_argument('--N', type=int, default=3, help="Num word substitutions used in attack")
    commandLineParser.add_argument('--B', type=int, default=100, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify learning rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_base_dir = args.TEST_DIR
    out_file = args.OUT
    classifier_out_file = args.CLASSIFIER_OUT
    layer_num = args.layer_num
    num_points_test = args.num_points_test
    num_points_val = args.num_points_val
    N = args.N
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    seed = args.seed
    cpu_use = args.cpu

    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/linear_classifier.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()
    
    # Load the model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Electra_Layer_Handler(model, layer_num=layer_num)
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

    # Load the test original and attacked data
    original_list, attack_list = load_test_adapted_data_sentences(test_base_dir, num_points_test)

    # Prepare input tensors
    original_comps = batched_get_layer_embedding(original_list, handler, tokenizer, device)

    attack_comps = batched_get_layer_embedding(attack_list, handler, tokenizer, device)

    labels = torch.LongTensor([0]*original_comps.size(0)+[1]*attack_comps.size(0))
    X = torch.cat((original_comps, attack_comps))

    # Shuffle all the data
    indices = torch.randperm(len(labels))
    labels = labels[indices]
    X = X[indices]

    # Split data
    X_val = X[:num_points_val]
    labels_val = labels[:num_points_val]
    X_train = X[num_points_val:]
    labels_train = labels[num_points_val:]   

    ds_train = TensorDataset(X_train, labels_train)
    ds_val = TensorDataset(X_val, labels_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    # Model
    model = LayerClassifier(X.size(-1))
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Create file
    with open(out_file, 'w') as f:
        text = f'Layer {layer_num}, N {N}\n'
        f.write(text)

    # Train
    for epoch in range(epochs):

        # train for one epoch
        text = '\n current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        with open(out_file, 'a') as f:
            f.write(text)
        print(text)
        train(dl_train, model, criterion, optimizer, epoch, device, out_file)

        # evaluate
        eval(dl_val, model, criterion, device, out_file)
    
    # Save the trained model for identifying adversarial attacks
    torch.save(model.state_dict(), classifier_out_file)
 