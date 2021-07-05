'''
Use training data to define chosen embedding space eigenvectors
For original and attacked test data determine the average size of the components in the eigenvector directions
Plot this against eigenvalue rank
'''

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tools import get_default_device
import matplotlib.pyplot as plt
from models import ElectraSequenceClassifier
from transformers import ElectraTokenizer
from layer_handler import Electra_Layer_Handler
from linear_pca_classifier import batched_get_layer_embedding, load_test_adapted_data_sentences

def stds(original, attack):
    '''
    original: Tensor [num_ranks x num_data_points]
    attack: Tensor [num_ranks x num_data_points]

    Return:
        The average (across rank) of the number of
        standard deviations between original and
        attack at each rank position

    This gives a measure of how out of distribution
    the attack is from the original
    '''
    with torch.no_grad():
        original_mean = torch.mean(original, dim=1)
        attack_mean = torch.mean(attack, dim=1)
        original_std = torch.std(original, dim=1)
        diff = torch.abs(attack_mean - original_mean)
        std_diff = diff/original_std
        return std_diff

def get_all_comps(X, eigenvectors, correction_mean):
    '''
    For each eigenvector (rank), calculates the
    magnitude of components in that direction for each
    data point

    Returns:
        Tensor [num_ranks, num_data_points]
    '''
    with torch.no_grad():
        # Correct by pre-calculated data mean
        X = X - correction_mean.repeat(X.size(0), 1)
        # Get every component in each eigenvector direction
        comps = torch.abs(torch.einsum('bi,ji->bj', X, eigenvectors))
    return torch.transpose(comps, 0, 1)

def get_avg_comps(X, eigenvectors, correction_mean):
    '''
    For each eigenvector, calculates average (across batch)
    magnitude of components in that direction
    '''
    with torch.no_grad():
        # Correct by pre-calculated data mean
        X = X - correction_mean.repeat(X.size(0), 1)
        # Get every component in each eigenvector direction
        comps = torch.einsum('bi,ji->bj', X, eigenvectors)
        # Get average of magnitude for each eigenvector rank
        avg_comps = torch.mean(torch.abs(comps), dim=0)
    return avg_comps


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DIR', type=str, help='attacked data base directory')
    commandLineParser.add_argument('EIGENVECTORS', type=str, help='Learnt eigenvectors .pt file for PCA projection')
    commandLineParser.add_argument('CORRECTION_MEAN', type=str, help='Learnt correction mean.pt file for PCA projection')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('--layer_num', type=int, default=12, help="Layer at which to use detector")
    commandLineParser.add_argument('--N', type=int, default=6, help="Number of words substituted")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    commandLineParser.add_argument('--num_points_test', type=int, default=2000, help="number of data points to use test")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    eigenvectors_path = args.EIGENVECTORS
    correction_mean_path = args.CORRECTION_MEAN
    out_file = args.OUT_FILE
    layer_num = args.layer_num
    N = args.N
    cpu_use = args.cpu
    num_points_test = args.num_points_test

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/average_comp_dist.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the Sentiment Classifier model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the eigenvectors for PCA decomposition and the correction mean
    eigenvectors = torch.load(eigenvectors_path)
    correction_mean = torch.load(correction_mean_path)

    # Create model handler for PCA layer detection check
    handler = Electra_Layer_Handler(model, layer_num=layer_num)
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

    # Load the test data
    original_list, attack_list = load_test_adapted_data_sentences(base_dir, num_points_test)
    print("Loaded data")

    # # Get embeddings
    # original_embeddings = batched_get_layer_embedding(original_list, handler, tokenizer, device)
    # attack_embeddings = batched_get_layer_embedding(attack_list, handler, tokenizer, device)

    # # Get average components against rank
    # original_avg_comps = get_avg_comps(original_embeddings, eigenvectors, correction_mean)
    # attack_avg_comps = get_avg_comps(attack_embeddings, eigenvectors, correction_mean)

    # # Plot the results
    # ranks = np.arange(len(original_avg_comps))
    # plt.plot(ranks, original_avg_comps, label='Original')
    # plt.plot(ranks, attack_avg_comps, label='Attacked')
    # plt.yscale('log')
    # plt.xlabel('Eigenvalue Rank')
    # plt.ylabel('Average Component Size')
    # plt.legend()
    # plt.savefig(out_file)
    # plt.clf()

    # # Report std diff between attack and original curves
    # original_comps = get_all_comps(original_embeddings, eigenvectors, correction_mean)
    # attack_comps = get_all_comps(attack_embeddings, eigenvectors, correction_mean)

    # std_diffs = stds(original_comps, attack_comps)
    # print("OOD metric", torch.mean(std_diffs))

    # # Plot std_diffs ranked by size
    # std_diffs_ordered, _ = torch.sort(std_diffs)
    # ranks = np.arange(len(std_diffs_ordered))
    
    # plt.plot(ranks, std_diffs_ordered)
    # plt.xlabel('std difference rank')
    # plt.ylabel('std difference')
    # plt.savefig('std_'+out_file)
    # plt.clf()

    # Determine error sizes in input embedding layer
        # - l2
        # - l-inf

    handler = Electra_Layer_Handler(model, layer_num=0)
    print(original_list[0])
    print()
    print(attack_list[0])

    # Get embeddings
    original_embeddings = batched_get_layer_embedding(original_list, handler, tokenizer, device)
    attack_embeddings = batched_get_layer_embedding(attack_list, handler, tokenizer, device)

    diffs = torch.reshape(torch.abs(attack_embeddings-original_embeddings), (attack_embeddings.size(0), -1))

    l2s = torch.sqrt(torch.sum(diffs**2, dim=1))
    print(f'l2: mean={torch.mean(l2s)} std={torch.std(l2s)}')

    linfs, _ = torch.max(diffs, dim=1)
    print(f'l-inf: mean={torch.mean(linfs)} std={torch.std(linfs)}')
