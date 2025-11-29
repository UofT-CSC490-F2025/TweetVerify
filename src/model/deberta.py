import torch
import torch.nn as nn
from transformers import DebertaV2PreTrainedModel, DebertaV2Model


class DebertaV3(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.deberta = DebertaV2Model(config)

        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_labels),
        )

        self.post_init()

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        cls_repr = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_repr)
        logits = self.classifier(x)
        return logits

    def get_name(self):
        return "deberta"
