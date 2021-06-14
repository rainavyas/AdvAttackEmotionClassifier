'''
Read in prepared original and substitution attack sentences.
Calculate the fooling rate.
Fooling Rate is calculated using only samples where the original
sentence is classified correctly
'''

import json
import argparse
import os
import sys

def get_fooling_rate(data_list):
    '''
    data_list: list
        [item1, item2, ...]
        where,
            item: dict
                with keys:
                    sentence
                    updated sentence
                    original prob
                    updated prob
                    true label
    '''
    total_count = 0
    fool_count = 0

    for item in data_list:
        original_prob = item['original prob']
        updated_prob = item['updated prob']
        label = item['true label']

        original_pred = original_prob.index(max(original_prob))
        updated_pred = updated_prob.index(max(updated_prob))

        if original_pred != label:
            continue

        if updated_pred != original_pred:
            fool_count += 1
        total_count += 1

    fool_rate = fool_count/total_count
    print("Total Count:", total_count)
    return fool_rate

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DIR', type=str, help='data base directory with attack results')

    args = commandLineParser.parse_args()
    base_dir = args.DIR

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack_stats.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the data
    data_list = []
    for i in range(2000):
        fname = base_dir + '/'+str(i)+'.txt'
        try:
            with open(fname, 'r') as f:
                item = json.load(f)
            data_list.append(item)
        except:
            print("Failed to load", i)

    fooling_rate = get_fooling_rate(data_list)
    print()
    print("Fooling Rate", fooling_rate)
