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

    # Get embeddings
    original_embeddings = batched_get_layer_embedding(original_list, handler, tokenizer, device)
    attack_embeddings = batched_get_layer_embedding(attack_list, handler, tokenizer, device)

    # Get average components against rank
    original_avg_comps = get_avg_comps(original_embeddings, eigenvectors, correction_mean)
    attack_avg_comps = get_avg_comps(attack_embeddings, eigenvectors, correction_mean)

    # Plot the results
    ranks = np.arange(len(original_avg_comps))
    plt.plot(ranks, original_avg_comps, label='Original')
    plt.plot(ranks, attack_avg_comps, label='Attacked')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Average Component Size')
    plt.legend()
    plt.savefig(out_file)

