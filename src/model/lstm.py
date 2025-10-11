import torch
import torch.nn as nn
import numpy as np

class MyLSTM(nn.Module):
    def __init__(self, model_w2v, hidden_size, num_classes):
        super(MyLSTM, self).__init__()
        self.vocab_size = len(model_w2v.wv) + 1
        self.emb_size = model_w2v.vector_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        # Add an extra zero-initialized vector as the padding vector
        self.emb.weight.data.copy_(torch.from_numpy(np.vstack((np.zeros((1, self.emb_size)), model_w2v.wv.vectors))))
        self.lstm = nn.LSTM(self.emb_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        embedded = self.emb(X)
        outputs, (ht, ct) = self.lstm(embedded)
        output = self.dropout_layer(torch.cat((ht[-2], ht[-1]), dim=1))
        output = self.fc(output)
        return output

    def parameters(self):
        for name, param in self.named_parameters():
            if name != 'emb.weight':
                yield param
    def get_name(self):
        return 'lstm'
