#!/usr/bin/env python
# coding: utf-8

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.graph_convolution import GraphConvolution
from model.rnn import RNN
from model.attention import LabelWiseAttention


class ZAGRNN(Classifier):

    def __init__(self, dataset, config):

        assert config.label_embedding.dimension == config.ZAGRNN.gcn_in_features, \
            "label embedding dimension should be same as gcn input feature dimension"

        super(ZAGRNN, self).__init__(dataset, config)

        self.rnn = RNN(
            config.embedding.dimension, config.ZAGRNN.hidden_dimension,
            num_layers=config.ZAGRNN.num_layers, batch_first=True,
            bidirectional=config.ZAGRNN.bidirectional,
            rnn_type=config.ZAGRNN.rnn_type)

        self.label_wise_attention = LabelWiseAttention(
            feat_dim=config.ZAGRNN.hidden_dimension*2 if config.ZAGRNN.bidirectional else config.ZAGRNN.hidden_dimension,
            label_emb_dim=config.label_embedding.dimension,
            store_attention_score=config.ZAGRNN.store_attention_score)

        if config.ZAGRNN.use_gcn:
            self.gcn = torch.nn.ModuleList([
                GraphConvolution(
                    in_features=config.ZAGRNN.gcn_in_features,
                    out_features=config.ZAGRNN.gcn_hidden_features,
                    bias=True,
                    act=torch.relu_,
                    featureless=False,
                    dropout=config.ZAGRNN.gcn_dropout),
                GraphConvolution(
                    in_features=config.ZAGRNN.gcn_hidden_features,
                    out_features=config.ZAGRNN.gcn_out_features,
                    bias=True,
                    act=torch.relu_,
                    featureless=False,
                    dropout=config.ZAGRNN.gcn_dropout)
            ])

            self.doc_out_transform = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=config.ZAGRNN.hidden_dimension*2 if config.ZAGRNN.bidirectional else config.ZAGRNN.hidden_dimension,
                    out_features=config.ZAGRNN.gcn_in_features + config.ZAGRNN.gcn_out_features
                ),
                torch.nn.ReLU()
            )

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.label_wise_attention.parameters()})
        if self.config.ZAGRNN.use_gcn:
            params.append({'params': self.gcn.parameters()})
            params.append({'params': self.doc_out_transform.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
            length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        else:
            raise NotImplementedError
        doc_embedding, _ = self.rnn(embedding, length)

        label_repr = self.label_embedding(
            batch[cDataset.DOC_LABEL_ID].to(self.config.device))
        attentive_doc_embedding = self.label_wise_attention(doc_embedding, label_repr)

        if self.config.ZAGRNN.use_gcn:
            label_repr_gcn = label_repr
            for gcn_layer in self.gcn:
                label_repr_gcn = gcn_layer(label_repr_gcn, batch[cDataset.DOC_LABEL_RELATION].to(self.config.device))
            label_repr = torch.cat((label_repr, label_repr_gcn), dim=1)

            return torch.sum(self.doc_out_transform(attentive_doc_embedding) * label_repr, dim=-1)

        return torch.sum(attentive_doc_embedding * label_repr, dim=-1)
