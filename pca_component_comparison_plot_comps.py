'''
Creates PCA component plots (at specified layer) between every pair of 
axes from start to end specified

Currently plot is only produced for start and start+1 axis
'''
import sys
import os
import argparse
from layer_handler import Electra_Layer_Handler
from models import ElectraSequenceClassifier
from data_prep import get_train
from data_prep_sentences import get_test
import torch
import torch.nn as nn
from transformers import ElectraTokenizer
from pca_tools import get_covariance_matrix, get_e_v
import pandas as pd
import seaborn as sns
import matplotlib as plt

def get_pca_principal_components(eigenvectors, correction_mean, X, num_comps, start):
    '''
    Returns components in num_comps most principal directions
    Dim 0 of X should be the batch dimension
    '''
    comps = []
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        for i in range(start, start+num_comps):
            v = eigenvectors[i]
            comp = torch.einsum('bi,i->b', X, v) # project to pca axis
            comps.append(comp.tolist())
    return comps[:num_comps]

def get_layer_embedding(sentences_list, handler, tokenizer):
    '''
    Return the CLS token embedding at layer specified in handler
    [batch_size x 768]
    '''
    encoded_inputs = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    with torch.no_grad():
        layer_embeddings = handler.get_layern_outputs(ids, mask)
        CLS_embeddings = layer_embeddings[:,0,:].squeeze()
    return CLS_embeddings

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TRAIN_DATA_PATH', type=str, help='train data filepath')
    commandLineParser.add_argument('TEST_DATA_PATH', type=str, help='test data filepath')
    commandLineParser.add_argument('OUT', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="layer to perturb")
    commandLineParser.add_argument('--num_points_train', type=int, default=6000, help="number train data points")
    commandLineParser.add_argument('--num_points_test', type=int, default=2000, help="number test data points")
    commandLineParser.add_argument('--num_comps', type=int, default=2, help="number of PCA components - fixed to 2 for now")
    commandLineParser.add_argument('--start', type=int, default=0, help="start of PCA components")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_data_path = args.TRAIN_DATA_PATH
    test_data_path = args.TEST_DATA_PATH
    out_file = args.OUT
    layer_num = args.layer_num
    num_points_train = args.num_points_train
    num_points_test = args.num_points_test
    num_comps = args.num_comps
    start = args.start

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pca_component_comparison_plot_comps.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Electra_Layer_Handler(model, layer_num=layer_num)
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

    # Use training data to get eigenvector basis for CLS token at correct layer
    input_ids, mask, _ = get_train('electra', train_data_path)
    indices = torch.randperm(len(input_ids))[:num_points_train]
    input_ids = input_ids[indices]
    mask = mask[indices]
    with torch.no_grad():
        layer_embeddings = handler.get_layern_outputs(input_ids, mask)
        CLS_embeddings = layer_embeddings[:,0,:].squeeze()
        correction_mean = torch.mean(CLS_embeddings, dim=0)
        cov = get_covariance_matrix(CLS_embeddings)
        e, v = get_e_v(cov)
    
    # Load the test data
    tweets_list, labels = get_test('electra', test_data_path)

    # Project embeddings to pca components
    embeddings = get_layer_embedding(tweets_list, handler, tokenizer)
    pca_comps = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    # Plot the data

    df = pd.DataFrame({"PCA "+str(start):pca_comps[start], "PCA "+str(start+1):pca_comps[start+1], "grade":labels})
    sns.set_theme(style="whitegrid")
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    sns_plot = sns.scatterplot(
                            data=df,
                            x="PCA "+str(start),
                            y="PCA "+str(start+1),
                            hue="grade",
                            palette=cmap)


    sns_plot.figure.savefig(out_file)