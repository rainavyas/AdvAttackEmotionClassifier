'''
Use a validation set to build a corpus of word frequencies
as part of the FGWS algorithm
'''

import sys
import os
import argparse
import json
from collections import defaultdict

dirname, _ = os.path.split(os.path.abspath(__file__))
sys.path.append(dirname+'/..')
from data_prep_sentences import get_train


if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA_PATH', type=str, help='train data filepath')
    commandLineParser.add_argument('OUT', type=str, help='json file to save frequencies dict to')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/learn_frequencies.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load sentences
    sentences, _ = get_train('electra', args.DATA_PATH)
    sentences = [s.split() for s in sentences]

    # Get frequencies
    frequencies = defaultdict(int)
    for sen in sentences:
        for word in sen:
            frequencies[word] += 1
    
    # Save dict
    with open(args.OUT, 'w') as f:
        json.dump(frequencies, f)
