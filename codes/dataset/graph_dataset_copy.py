#!/usr/bin/env python
# coding: utf-8

import torch
import torch.sparse
import numpy as np
import scipy.sparse as sp
from dataset.classification_dataset import ClassificationDataset as cDataset
from util import Logger


class GraphDataSet:

    def __init__(self, adj_path, node_emb_path, conf):
        assert isinstance(adj_path, list), "Adjancency path should be list"
        assert isinstance(node_emb_path, str), "Node embedding path should be str"

        self.adj_path = adj_path
        self.node_emb_path = node_emb_path
        self.conf = conf
        self.logger = Logger(conf)
        self._load_node_emb()
        self._load_adj()

    def _load_adj(self):
        
        order = []
        if self.conf.data.label_map_of_relation_files:
            self.logger.info("this step is necessary to re-order the adj matrix")
            # get source label map
            source_label_map = dict()
            with open(self.conf.data.label_map_of_relation_files, mode='r') as fin:
                for line in fin:
                    # star from idx 0
                    label, idx = line.strip('\n').split("\t")
                    source_label_map[label] = int(idx)

            empty_dataset = cDataset(self.conf, [], mode='train')
            target_id_to_label_map = empty_dataset.id_to_label_map
            for i in range(len(target_id_to_label_map)):
                label = target_id_to_label_map[i]
                order.append(source_label_map[label])

        def rearrange(_np_mat, _order):
            if _order:
                return _np_mat[np.ix_(_order, _order)]
            return _np_mat

        if len(self.adj_path) == 1:
            self.adj = torch.from_numpy(
                rearrange(self.normalize(
                    sp.load_npz(self.adj_path[0]).astype(np.float32)).todense(), order))
        else:
            self.adj = torch.stack([
                torch.from_numpy(
                    rearrange(self.normalize(sp.load_npz(path).astype(np.float32)).todense(), order))
                for path in self.adj_path])

    def _load_node_emb(self):
        if not self.node_emb_path:
            self.node_emb = None
        else:
            raise NotImplementedError

    @staticmethod
    def normalize(mx, method="inv"):
        """same as Rethinking knowledge graph propagation for zero-shot learning
        https://github.com/cyvius96/DGP/blob/master/utils.py"""

        if method == "inv":
            rowsum = np.array(mx.sum(0))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
        if method == "sys":
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -0.5).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx

    @staticmethod
    def spm_to_spt(sparse_mx):
        """same as Rethinking knowledge graph propagation for zero-shot learning
        https://github.com/cyvius96/DGP/blob/master/utils.py"""
        sparse_mx = sparse_mx.tocoo()
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
