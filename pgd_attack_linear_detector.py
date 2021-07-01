'''
Perform PGD attack in first embedding layer
Visualize embedding space in PCA decomposition
Train linear detector on it
'''
import torch
import torch.nn as nn
import sys
import os
import argparse
from models import ElectraSequenceClassifier
from layer_handler import Electra_Layer_Handler
from data_prep_sentences import get_test
from transformers import ElectraTokenizer
from tools import accuracy_topk

class Attack(torch.nn.Module):
  def __init__(self, attack_init):
    super(Attack, self).__init__()
    self.attack = torch.nn.Parameter(attack_init, requires_grad=True)

  def forward(self, X, attention_mask, layer_handler):
    X_attacked = X + self.attack
    y = layer_handler.pass_through_rest(X_attacked, attention_mask)
    return y

def clip_params(model, epsilon):
    old_params = {}

    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    old_params['attack'][old_params['attack']>epsilon] = epsilon
    old_params['attack'][old_params['attack']<(-1*epsilon)] = -1*epsilon
 
    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])

def train_pgd(X, attention_mask, target, attack_model, criterion, optimizer, epoch, epsilon, layer_handler):
    """
        Run one train epoch
    """
    attack_model.train()
    X_var = X.to(torch.device('cpu')) 
    attention_mask_var = attention_mask.to(torch.device('cpu'))
    target_var = target.to(torch.device('cpu'))

    # compute output
    output = attack_model(X_var, attention_mask_var, layer_handler)
    loss = criterion(output, target_var)
    loss_neg = -1*loss

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss_neg.backward()
    optimizer.step()

    # clip the parameters
    clip_params(attack_model, epsilon)

    output = output.float()
    loss = loss_neg.float()

    # measure accuracy
    prec1 = accuracy_topk(output.data, target_var)

    print(f'Epoch: {epoch}\t Loss: {loss}\t Accuracy: {prec1}')

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

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DATA_PATH', type=str, help='data filepath')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="tweet index to start at")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help="tweet index to end at")
    commandLineParser.add_argument('--epsilon', type=float, default=0.01, help="l-inf pgd perturbation size")
    commandLineParser.add_argument('--lr', type=float, default=0.1, help="pgd learning rate")
    commandLineParser.add_argument('--epochs', type=int, default=20, help="Number of epochs for PGD attacks")
    commandLineParser.add_argument('--seed', type=int, default=1, help="seed for randomness")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_path = args.DATA_PATH
    start_ind = args.start_ind
    end_ind = args.end_ind
    epsilon = args.epsilon
    lr = args.lr
    epochs = args.epochs
    seed = args.seed

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pgd_attack_linear_detector.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Set seed
    torch.manual_seed(seed)

    # Load the model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() 

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

    # Create attack model
    attack_init = torch.zeros_like(input_embeddings)
    attack_model = Attack(attack_init)

    # Perform PGD attack
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_pgd(input_embeddings, mask, labels, attack_model, criterion, optimizer, epoch, epsilon, handler)
    eval_pgd(input_embeddings, mask, labels, attack_model, criterion, handler)


    # ---------------------
    # 2) Visualize PCA embedding space decomposition
    # ---------------------






