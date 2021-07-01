import torch
import torch.nn as nn

class Electra_Layer_Handler():
    '''
    Allows you to get the outputs at a Electra layer of a trained Electra based classifier

    AND has a separate method to pass an embedding through
    and remaining layers of the model and further for ElectraClassifier

    Electra model layers 1-12
    layer 0 gives input embeddings based on input_ids
    '''

    def __init__(self, trained_model, layer_num=1):
        trained_model.eval()
        self.model = trained_model
        self.layer_num = layer_num

    def get_layern_outputs(self, input_ids, attention_mask, device=torch.device('cpu')):
        '''
        Get output hidden states from nth layer
        '''
        self.model.to(device)
        
        # Need to extend mask for encoder - from HuggingFace implementation
        self.input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.model.electra.get_extended_attention_mask(attention_mask, self.input_shape, device)

        hidden_states = self.model.electra.embeddings(input_ids=input_ids)
        for layer_module in self.model.electra.encoder.layer[:self.layer_num]:
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states

    def pass_through_rest(self, hidden_states, attention_mask, device=torch.device('cpu')):
        '''
        Pass hidden states through remainder of ElectraClassifier model
        after nth layer
        '''

        extended_attention_mask: torch.Tensor = self.model.electra.get_extended_attention_mask(attention_mask, self.input_shape, device)

        for layer_module in self.model.electra.encoder.layer[self.layer_num:]:
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]

        logits = self.model.classifier(hidden_states)
        return logits
    
    def pass_through_some(self, hidden_states, attention_mask, output_layer=12, device=torch.device('cpu')):
        '''
        Same as pass_through_rest function but only passes
        up to specified layer_number
        '''
        extended_attention_mask: torch.Tensor = self.model.electra.get_extended_attention_mask(attention_mask, self.input_shape, device)

        for layer_module in self.model.electra.encoder.layer[self.layer_num:output_layer]:
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states