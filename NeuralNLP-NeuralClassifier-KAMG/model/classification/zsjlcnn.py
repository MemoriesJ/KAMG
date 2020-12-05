#!/usr/bin/env python
# coding: utf-8

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.graph_convolution import MultiGraphConvolution
from model.attention import LabelWiseAttention
from model.fusion import Fusion, FusionType


class ZSJLCNN(Classifier):

    def __init__(self, dataset, config):

        assert config.label_embedding.dimension == config.ZSJLCNN.gcn_in_features, \
            "label embedding dimension should be same as gcn input feature dimension"
        assert len(config.data.label_relation_files) >= 2, \
            "this model should utilize at least 2 different graphs' adjacency"

        super(ZSJLCNN, self).__init__(dataset, config)

        self.conv = torch.nn.Conv1d(
            in_channels=config.embedding.dimension,
            out_channels=config.ZSJLCNN.num_kernels,
            kernel_size=config.ZSJLCNN.kernel_size,
            padding=int(config.ZSJLCNN.kernel_size // 2))

        self.label_wise_attention = LabelWiseAttention(
            feat_dim=config.ZSJLCNN.num_kernels,
            label_emb_dim=config.label_embedding.dimension,
            store_attention_score=config.ZSJLCNN.store_attention_score)

        self.multi_gcn = torch.nn.ModuleList([
            MultiGraphConvolution(
                n_adj=len(config.data.label_relation_files),
                in_features=config.ZSJLCNN.gcn_in_features,
                out_features=config.ZSJLCNN.gcn_hidden_features,
                bias=True,
                act=torch.relu_,
                featureless=False,
                dropout=config.ZSJLCNN.gcn_dropout),
            MultiGraphConvolution(
                n_adj=len(config.data.label_relation_files),
                in_features=config.ZSJLCNN.gcn_hidden_features,
                out_features=config.ZSJLCNN.gcn_out_features,
                bias=True,
                act=torch.relu_,
                featureless=False,
                dropout=config.ZSJLCNN.gcn_dropout)
        ])

        self.multi_gcn_fuse = Fusion(config)

        if config.fusion.fusion_type == FusionType.CONCATENATION:
            out_tmp = config.ZSJLCNN.gcn_in_features + config.fusion.out_features
        elif config.fusion.fusion_type == FusionType.ATTACH:
            out_tmp = config.ZSJLCNN.gcn_in_features + config.ZSJLCNN.gcn_out_features * 2
        else:
            raise NotImplementedError
        self.doc_out_transform = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=config.ZSJLCNN.num_kernels,
                out_features=out_tmp
            ),
            torch.nn.ReLU()
        )

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.conv.parameters()})
        params.append({'params': self.label_wise_attention.parameters()})
        params.append({'params': self.multi_gcn.parameters()})
        params.append({'params': self.multi_gcn_fuse.parameters()})
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

        label_repr_gcn = torch.unsqueeze(label_repr, dim=0).repeat(len(self.config.data.label_relation_files), 1, 1)
        for gcn_layer in self.multi_gcn:
            label_repr_gcn = gcn_layer(label_repr_gcn, batch[cDataset.DOC_LABEL_RELATION].to(self.config.device))
        # do some other fusion operations
        label_repr_gcn = self.multi_gcn_fuse(torch.unbind(label_repr_gcn, dim=0))
        label_repr = torch.cat((label_repr, label_repr_gcn), dim=1)

        return torch.sum(self.doc_out_transform(attentive_doc_embedding) * label_repr, dim=-1)
