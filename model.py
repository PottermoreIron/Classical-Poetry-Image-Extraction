# code for the model
from transformers.models.albert.modeling_albert import *
from torch.nn.utils.rnn import pad_sequence
import torchcrf


class BertBiLstmCRF(AlbertPreTrainedModel):
    def __init__(self, config):
        super(BertBiLstmCRF, self).__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size // 2, batch_first=True,
                            num_layers=config.lstm_num_layers, dropout=config.dropout, bidirectional=True)
        self.linear_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = torchcrf.CRF(config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, data, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, ):
        # TODO
        pass
