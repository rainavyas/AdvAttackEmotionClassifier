'''
Batch application of textfooler attack
'''

import sys
import os
import argparse
from transformers import ElectraTokenizer
from models import ElectraSequenceClassifier
import textattack
import torch

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/textfooler_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load model and tokenizer -> model wrapper
    model = ElectraSequenceClassifier()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    model_wrapper = textattack.models.wrappers.ModelWrapper(model, tokenizer)

    # Create Textfooler object
    attack = textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019.build(model_wrapper)

    # Perform attack
    input_text = "I really enjoyed the new movie that came out last month."
    label = 1
    attack_result = attack.attack(input_text, label)
    print(attack_result)

