'''
Use upper bound saliency at the word embedding level to choose word to substitute
with synonym.

This is currently designed for Electra encoder based models only
'''

import torch
import torch.nn as nn
import nltk
from nltk.corpus import wordnet as wn
from layer_handler import Electra_Layer_Handler
from models import ElectraSequenceClassifier
from data_prep_sentences import get_test
import json
from transformers import ElectraTokenizer
import sys
import os
import argparse
from collections import OrderedDict

def get_token_saliencies(sentence, label, handler, criterion, tokenizer):
    '''
    Returns tensor of saliencies in token order

    Saliency is an upperbound saliency, given by the size of the vector of the
    loss functions derivative wrt to the word embedding.
    Word embeddings are taken from the input embedding layer before the encoder
    Note that the label should be the true label (1 or 0)
    '''

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    target = torch.LongTensor([label])

    embeddings = handler.get_layern_outputs(ids, mask)
    embeddings.retain_grad()
    logits = handler.pass_through_rest(embeddings, mask)
    loss = criterion(logits, target)

    # Determine embedding token saliencies
    loss.backward()
    embedding_grads = embeddings.grad
    saliencies = torch.linalg.norm(embedding_grads, dim=-1).squeeze()

    return saliencies


def attack_sentence(sentence, label, model, handler, criterion, tokenizer, max_syn=5, N=1):
    '''
    Identifies the N most salient words (by upper bound saliency)
    Finds synonyms for these words using WordNet
    Selects the best synonym to replace with based on Forward Pass to maximise
    the loss function, sequentially starting with most salient word

    Returns the original_sentence, updated_sentence, original_logits, updated_logits
    '''
    model.eval()

    token_saliencies = get_token_saliencies(sentence, label, handler, criterion, tokenizer)
    token_saliencies[0] = 0
    token_saliencies[-1] = 0

    inds = torch.argsort(token_saliencies, descending=True)
    if len(inds) > N:
        inds = inds[:N]

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids'].squeeze()
    mask = encoded_inputs['attention_mask']

    assert len(token_saliencies) == len(ids), "tokens and saliencies mismatch"

    for i, ind in enumerate(inds):
        target_id = ids[ind]
        word_token = tokenizer.convert_ids_to_tokens(target_id.item())

        synonyms = []
        for syn in wn.synsets(word_token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if len(synonyms)==0:
            # print("No synonyms for ", word_token)
            updated_logits = model(torch.unsqueeze(ids, dim=0), mask).squeeze()
            if i==0:
                original_logits = updated_logits.clone()
            continue

        # Remove duplicates
        synonyms = list(OrderedDict.fromkeys(synonyms))

        if len(synonyms) > max_syn+1:
            synonyms = synonyms[:max_syn+1]

        best = (target_id, 0) # (id, loss)
        for j, syn in enumerate(synonyms):
            try:
                new_id = tokenizer.convert_tokens_to_ids(syn)
            except:
                print(syn+" is not a token")
                continue

            ids[ind] = new_id
            with torch.no_grad():
                logits = model(torch.unsqueeze(ids, dim=0), mask)
                loss = criterion(logits, torch.LongTensor([label])).item()

            if i==0 and j==0:
                original_logits = logits.squeeze()
            if loss > best[1]:
                best = (new_id, loss)
                updated_logits = logits.squeeze()
        ids[ind] = best[0]

    updated_sentence = tokenizer.decode(ids)
    updated_sentence = updated_sentence.replace('[CLS] ', '')
    updated_sentence = updated_sentence.replace(' [SEP]', '')
    updated_sentence = updated_sentence.replace('[UNK]', '')

    return sentence, updated_sentence, original_logits, updated_logits

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('DATA_PATH', type=str, help='data filepath')
    commandLineParser.add_argument('--max_syn', type=int, default=10, help="Number of synonyms to search")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words to substitute")
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="tweet index to start at")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help="tweet index to end at")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_path = args.DATA_PATH
    max_syn = args.max_syn
    N = args.N
    start_ind = args.start_ind
    end_ind = args.end_ind

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/upper_bound_saliency_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    nltk.download('wordnet')

    # Load the model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Electra_Layer_Handler(model, layer_num=0)

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=0)

    # Create directory to save files in
    dir_name = 'Attacked_Data_N'+str(N)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Get all data
    tweets_list, labels = get_test('electra', data_path)

    for ind in range(start_ind, end_ind):

        # Get the relevant data
        sentence = tweets_list[ind]
        label = labels[ind]

        # Attack and save the sentence
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(sentence, label, model, handler, criterion, tokenizer, max_syn=max_syn, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/'+str(ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))
