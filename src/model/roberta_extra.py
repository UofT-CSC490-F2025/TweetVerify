import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel


class Roberta_Extra(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2

        self.num_extra_features = 2

        self.roberta = RobertaModel(config)

        hidden_size = config.hidden_size

        combined_size = hidden_size + self.num_extra_features

        self.batch_norm = nn.BatchNorm1d(self.num_extra_features)

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_labels),
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        extra_features=None,
        labels=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        cls_repr = outputs.last_hidden_state[:, 0, :]

        if extra_features is not None:

            extra_features = extra_features.to(cls_repr.dtype)

            norm_features = self.batch_norm(extra_features)

            x = torch.cat((cls_repr, norm_features), dim=1)
        else:

            zeros = torch.zeros(cls_repr.size(0), self.num_extra_features).to(
                cls_repr.device, dtype=cls_repr.dtype
            )
            x = torch.cat((cls_repr, zeros), dim=1)

        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def get_name(self):
        return "roberta_extra"
