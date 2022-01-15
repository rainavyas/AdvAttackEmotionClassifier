'''
Batch application of textfooler or BAE attack
'''

import sys
import os
import argparse
from transformers import ElectraTokenizer
from models import ElectraSequenceClassifier
import textattack
import torch
import torch.nn as nn
from model_wrapper import PyTorchModelWrapper
from data_prep_sentences import get_test
import json

def attack_sentence(sentence, label, attack, model, tokenizer):
    '''
    Apply the attack
    '''
    # Get attack sentence
    attack_result = attack.attack(sentence, label)
    updated_sentence = attack_result.perturbed_text()

    # Get original probabilities
    softmax = nn.Softmax(dim=0)
    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    with torch.no_grad():
        logits = model(ids).squeeze()
        original_probs = softmax(logits).tolist()

    # Get updated probabilities
    softmax = nn.Softmax(dim=0)
    encoded_inputs = tokenizer([updated_sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    with torch.no_grad():
        logits = model(ids).squeeze()
        updated_probs = softmax(logits).tolist()
    
    return sentence, updated_sentence, original_probs, updated_probs

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('DATA_PATH', type=str, help='data filepath')
    commandLineParser.add_argument('ATTACK_TYPE', type=str, choices=['BAE', 'textfooler'], help='BAE or textfooler attack')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="tweet index to start at")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help="tweet index to end at")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/textattack_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load model and tokenizer -> model wrapper
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    # model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)
    model_wrapper = PyTorchModelWrapper(model, tokenizer)

    # Create Textfooler object
    if args.ATTACK_TYPE == 'BAE':
        attack = textattack.attack_recipes.bae_garg_2019.BAEGarg2019.build(model_wrapper)
    elif args.ATTACK_TYPE == 'textfooler':
        attack = textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019.build(model_wrapper)
    else:
        raise TypeError("Incorrect attack type")

    # Create directory to save files in
    dir_name = f'{args.ATTACK_TYPE}_Attacked_Data'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Get all data
    tweets_list, labels = get_test('electra', args.DATA_PATH)

    for ind in range(args.start_ind, args.end_ind):
        print(f'On {ind}/{args.end_ind}')

        # Get the relevant data
        sentence = tweets_list[ind]
        label = labels[ind]

        # Attack and save the sentence
        sentence, updated_sentence, original_probs, updated_probs = attack_sentence(sentence, label, attack, model, tokenizer)
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/'+str(ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))

