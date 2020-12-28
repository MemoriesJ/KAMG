#!/usr/bin/env python
# coding: utf-8

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.graph_convolution import GraphConvolution
from model.attention import LabelWiseAttention


class ZAGCNN(Classifier):

    def __init__(self, dataset, config):

        super(ZAGCNN, self).__init__(dataset, config)

        self.conv = torch.nn.Conv1d(
            in_channels=config.embedding.dimension,
            out_channels=config.ZAGCNN.num_kernels,
            kernel_size=config.ZAGCNN.kernel_size,
            padding=int(config.ZAGCNN.kernel_size // 2))

        self.label_wise_attention = LabelWiseAttention(
            feat_dim=config.ZAGCNN.num_kernels,
            label_emb_dim=config.label_embedding.dimension,
            store_attention_score=config.ZAGCNN.store_attention_score)

        if config.ZAGCNN.use_gcn:

            assert config.label_embedding.dimension == config.ZAGCNN.gcn_in_features, \
                "label embedding dimension should be same as gcn input feature dimension"

            self.gcn = torch.nn.ModuleList([
                GraphConvolution(
                    in_features=config.ZAGCNN.gcn_in_features,
                    out_features=config.ZAGCNN.gcn_hidden_features,
                    bias=True,
                    act=torch.relu_,
                    featureless=False,
                    dropout=config.ZAGCNN.gcn_dropout),
                GraphConvolution(
                    in_features=config.ZAGCNN.gcn_hidden_features,
                    out_features=config.ZAGCNN.gcn_out_features,
                    bias=True,
                    act=torch.relu_,
                    featureless=False,
                    dropout=config.ZAGCNN.gcn_dropout)
            ])

            self.doc_out_transform = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=config.ZAGCNN.num_kernels,
                    out_features=config.ZAGCNN.gcn_in_features + config.ZAGCNN.gcn_out_features
                ),
                torch.nn.ReLU()
            )

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.conv.parameters()})
        params.append({'params': self.label_wise_attention.parameters()})
        if self.config.ZAGCNN.use_gcn:
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
        else:
            raise NotImplementedError
        embedding = embedding.transpose(1, 2)
        doc_embedding = torch.relu_(self.conv(embedding))
        doc_embedding = doc_embedding.transpose(1, 2)

        label_repr = self.label_embedding(
            batch[cDataset.DOC_LABEL_ID].to(self.config.device))
        attentive_doc_embedding = self.label_wise_attention(doc_embedding, label_repr)

        if self.config.ZAGCNN.use_gcn:
            label_repr_gcn = label_repr
            for gcn_layer in self.gcn:
                label_repr_gcn = gcn_layer(label_repr_gcn, batch[cDataset.DOC_LABEL_RELATION].to(self.config.device))
            label_repr = torch.cat((label_repr, label_repr_gcn), dim=1)

            return torch.sum(self.doc_out_transform(attentive_doc_embedding) * label_repr, dim=-1)

        return torch.sum(attentive_doc_embedding * label_repr, dim=-1)
