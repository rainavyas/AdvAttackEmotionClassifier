'''
Greedy N-word concatenation attack
'''

import torch
import torch.nn as nn
from models import ElectraSequenceClassifier
from data_prep_sentences import get_test
import json
from transformers import ElectraTokenizer
import sys
import os
import argparse
import json

def attack_sentence(sentence, label, model, criterion, tokenizer, word_list, search_size=2000, N=6):
    '''
    Perform N word greedy concatenation attack
    Returns the original_sentence, updated_sentence, original_logits, updated_logits
    '''

    model.eval()

    attack_sentence = sentence[:]
    for n in range(N):
        # Find nth attack word to concatenate
        best = ['', 0] # [word, loss]
        for word in word_list[:search_size]:
            new_sentence = attack_sentence + ' ' + word
            encoded_inputs = tokenizer([new_sentence], padding=True, truncation=True, return_tensors="pt")
            ids = encoded_inputs['input_ids']
            mask = encoded_inputs['attention_mask']
            logits = model(ids, mask)
            loss = criterion(logits, torch.LongTensor([label])).item()

            if loss > best[1]:
                best[0] = word
                best[1] = loss
                updated_logits = logits.squeeze()
        attack_sentence = attack_sentence + ' ' + best[0]
    
    # Get original logits
    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    original_logits = model(ids, mask)

    return sentence, attack_sentence, original_logits.squeeze(), updated_logits


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('DATA_PATH', type=str, help='data filepath')
    commandLineParser.add_argument('VOCAB', type=str, help='word list to search path')
    commandLineParser.add_argument('--search_size', type=int, default=2000, help="Number of words to search")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words to concatenate")
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="tweet index to start at")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help="tweet index to end at")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_path = args.DATA_PATH
    vocab_file = args.VOCAB
    search_size = args.search_size
    N = args.N
    start_ind = args.start_ind
    end_ind = args.end_ind

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/concatenation_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()


    # Get list of words to try
    with open(vocab_file, 'r') as f:
        test_words = json.loads(f.read())
    test_words = [str(word).lower() for word in test_words]


    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=0)

    # Create directory to save files in
    dir_name = 'Concatenation_Attacked_Data_N'+str(N)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Get all data
    tweets_list, labels = get_test('electra', data_path)

    for ind in range(start_ind, end_ind):

        # Get the relevant data
        sentence = tweets_list[ind]
        label = labels[ind]

        # Attack and save the sentence
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(sentence, label, model, criterion, tokenizer, test_words, search_size=search_size, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/'+str(ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))
