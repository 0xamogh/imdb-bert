import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        #Defining the bert model
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        #current bert base has 768 output features --> 1 because current problem is a binary classification problem
        self.out = nn.Linear(768, 1)
    
    #parameters are the inputs to the forward path of the neural network
    # mask = attention masks of the model
    def forward(self, ids, mask, token_type_ids):
        # parameters that bert by transformers takes in
        last_hidden_states, bert_pooler_output = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        bert = self.bert_drop(bert_pooler_output)
        output = self.out(bert)
        return output    