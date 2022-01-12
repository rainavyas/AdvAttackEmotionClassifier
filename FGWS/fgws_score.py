'''
Get the change in probability after frequency based substitution
Calculate the best f1 score using this approach
'''

import sys
import os
import argparse
import json
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import ElectraTokenizer
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.metrics import precision_recall_curve

dirname, _ = os.path.split(os.path.abspath(__file__))
sys.path.append(dirname+'/..')
from data_prep_sentences import get_test
from tools import get_default_device
from models import ElectraSequenceClassifier
from linear_pca_classifier import load_test_adapted_data_sentences

def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precisions*(beta**2))+recalls))
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]

def substitute(word, freq_dict):
    '''
    Find synonym of word with a higher frequency
    '''
    best = (word, freq_dict[word])
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if freq_dict[lemma.name()] > best[1]:
                best = (lemma.name(), freq_dict[lemma.name()])
    return best[0]

def model_pred(sentence, model, tokenizer):
    '''
    Return probability of model prediction label
    '''
    encoded_inputs = tokenizer([sentence], return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    logits = model(ids, mask).squeeze()
    m = nn.Softmax(dim=0)
    probs = m(logits)
    probs = probs.cpu().tolist()
    return max(probs)

def get_score(X, freq_dict, model, tokenizer, delta=1):
    '''
    Calculate FGWS score:
    Identify low frequency words in sentence X
    Substitute these words with higher frequency synonyms (create X')
    Calculate change in output probability from model, i.e. f(X) - f(X')
    '''
    word_list = X.split()
    X_dash_words = []
    for word in word_list:
        if freq_dict[word] < delta:
            X_dash_words.append(substitute(word, freq_dict))
        else:
            X_dash_words.append(word)
    X_dash = ' '.join(X_dash_words)
    f_X = model_pred(X, model, tokenizer)
    f_X_dash = model_pred(X_dash, model, tokenizer)
    return f_X - f_X_dash
    

if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('ATTACKED_DATA', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('MODEL', type=str, help='model filepath')
    commandLineParser.add_argument('FREQ', type=str, help='json frequency dict filepath')
    commandLineParser.add_argument('--cpu', type=str, default='yes', help="force cpu use")
    commandLineParser.add_argument('--num_points_test', type=int, default=2000, help="number of data points to use test")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/fgws_score.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if args.cpu == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()
    
    # Set tuning constant
    DELTA = 1 # Substitute words with frequencies lower than this

    # Load frequency dictionary
    with open(args.FREQ) as f:
        freq_dict = json.load(f)
    freq_dict = defaultdict(int, freq_dict)
    print('Got frequency dict')

    # Load the test original and attacked data
    original_list, attack_list = load_test_adapted_data_sentences(args.ATTACKED_DATA, args.num_points_test)
    print("Loaded test data")

    # Load model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()
    print("Loaded model")
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

    # Get FGWS scores
    original_scores = []
    attack_scores = []
    for o,a in zip(original_list, attack_list):
        original_scores.append(get_score(o, freq_dict, model, tokenizer, delta=DELTA))
        attack_scores.append(get_score(a, freq_dict, model, tokenizer, delta=DELTA))
    
    # Calculate Best F1 score
    labels = [0]*len(original_scores) + [1]*len(attack_scores)
    scores = original_scores + attack_scores
    precision, recall, _ = precision_recall_curve(labels, scores)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print(f"Best F1 Score: {best_f1}") 
    


