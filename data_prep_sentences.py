'''
Same as data_prep.py, but instead of preparing tensors, returns
a list of the raw sentences and associated labels
'''
from data_prep import read_file

def get_data(filepath, arch):
    allowed_arch = ['electra', 'bert', 'roberta']
    if arch not in allowed_arch:
        raise Exception('Invalid architecture, only allowed: electra, bert, roberta')
    
    CLASS_TO_IND = {
        'love': 0,
        'joy': 1,
        'fear': 2,
        'anger': 3,
        'surprise': 4,
        'sadness': 5,
    }

    tweets_list, labels = read_file(filepath, CLASS_TO_IND)
    return tweets_list, labels

def get_train(arch, filepath='../data/train.txt'):
    return get_data(filepath, arch)

def get_val(arch, filepath='../data/val.txt'):
    return get_data(filepath, arch)

def get_test(arch, filepath='../data/test.txt'):
    return get_data(filepath, arch)