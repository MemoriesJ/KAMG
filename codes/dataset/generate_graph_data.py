#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import json
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent_path, root_path = file.parent, file.parents[1]
sys.path.append(str(root_path))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent_path))
except ValueError: # Already removed
    pass

from config import Config


def get_hier_adj(parent_child_map, label_map):

    adj = np.zeros((len(label_map), len(label_map)))
    for parent in parent_child_map:
        adj[label_map[parent], [label_map[child] for child in parent_child_map[parent]]] = 1

    adj = np.maximum(adj, adj.T).astype(np.int)
    adj = sp.coo_matrix(adj)
    adj.setdiag(1)

    return adj


def get_cooc_corr_adj(reverse_label_map, config, cooc_threshold, tau=0.4, p=0.2):

    # binarize the training labels for get corr adj
    train_labels = []
    for f in config.data.train_json_files:
        with open(f) as fin:
            for json_str in fin:
                train_labels.append(json.loads(json_str)["doc_label"])

    full_labels = [reverse_label_map[i] for i in range(len(reverse_label_map))]
    mlb = MultiLabelBinarizer(classes=full_labels)
    binary_train_labels = mlb.fit_transform(train_labels)
    binary_train_labels = sp.coo_matrix(binary_train_labels)

    oc_cooc = binary_train_labels.transpose().dot(binary_train_labels).astype(np.float32)
    oc = oc_cooc.diagonal()
    corr = oc_cooc / oc

    cooc_adj = np.asarray(oc_cooc.todense())
    np.fill_diagonal(cooc_adj, 0)
    idx = int((cooc_adj.shape[0] ** 2) * cooc_threshold)
    threshold_value = np.sort(cooc_adj.flatten())[-idx]
    cooc_adj = (cooc_adj > threshold_value).astype(np.int)
    cooc_adj = sp.coo_matrix(cooc_adj)
    cooc_adj.setdiag(1)

    # correlation graph
    corr = np.asarray(corr)
    corr[np.isnan(corr)] = 0.
    corr[np.isinf(corr)] = 0.
    corr_adj = (corr >= tau).astype(np.float32)

    # corr over smooth
    np.fill_diagonal(corr_adj, 0.)
    corr_adj = p / np.sum(corr_adj, axis=1, keepdims=True)
    corr_adj[np.isnan(corr_adj)] = 0.
    corr_adj[np.isinf(corr_adj)] = 0.
    corr_adj = np.repeat(corr_adj, corr.shape[0], axis=1)
    np.fill_diagonal(corr_adj, 1-p)
    corr_adj = sp.coo_matrix(corr_adj)

    return cooc_adj, corr_adj


def get_simi_adj(sorted_label_emb, simi_threshold):
    
    # https://icml2020-submission.github.io/MAGCN/
    similarity = 1 - pdist(sorted_label_emb, 'cosine')
    similarity[np.isnan(similarity)] = 0.

    idx = int((sorted_label_emb.shape[0]**2) * simi_threshold / 2)
    threshold_value = np.sort(similarity.flatten())[-idx]
    similarity[similarity > threshold_value] = 1.
    similarity[similarity <= threshold_value] = 0.

    # similarity[similarity > simi_threshold] = 1
    # similarity[similarity <= simi_threshold] = 0

    adj = squareform(similarity).astype(np.int)
    adj = sp.coo_matrix(adj)
    adj.setdiag(1)

    return adj


def impose_adjs(adj1, adj2, lamb=0.4, tau=0.02, eta=0.4):

    adj = (1-lamb) * adj1 + lamb * adj2
    # adj = (adj >= tau).astype(np.int)
    adj[adj < tau] = 0.
    adj = eta * adj + (1-eta) * np.identity(adj.shape[0])
    adj = sp.coo_matrix(adj)

    return adj


def main(conf, args):

    # get the label map / reverse label map / parent child map
    # only labels appear in train/validate/test will be stored
    used_labels = set()
    for f in conf.data.train_json_files + conf.data.validate_json_files + conf.data.test_json_files:
        with open(f) as fin:
            for line in fin:
                used_labels = used_labels.union(set(json.loads(line)["doc_label"]))

    label_map = {}
    id2label = {}
    parent_child_map = {}
    with open(conf.task_info.hierar_taxonomy) as fin:
        for line in fin:
            parent = None
            for label in line.strip('\n').split('\t'):
                if label == 'Root':
                    break
                if label in used_labels:
                    if label not in label_map:
                        label_map[label] = len(label_map)
                        id2label[len(id2label)] = label
                    if parent is None:
                        parent_child_map[label] = []
                        parent = label
                        continue
                    else:
                        parent_child_map[parent].append(label)

    # get original label map
    print("saving label map of adjacency files. ")
    with open(conf.data.label_map_of_relation_files, mode='w') as fout:
        for k in label_map:
            fout.write("{}\t{}\n".format(k, label_map[k]))

    # load pretrained token embedding
    emb = {}
    with open(conf.feature.token_pretrained_file) as fin:
        for line in fin:
            data = line.strip().split(' ')
            # Check embedding info
            if len(data) == 2:
                assert int(data[1]) == conf.label_embedding.dimension, \
                    "Pretrained embedding dim not matching: %s, %d" % (
                        data[1], conf.label_embedding.dimension)
                continue
            emb[data[0]] = np.array([float(i) for i in data[1:]], dtype=np.float32)

    # get label embedding
    label_emb = {}
    label_tokens = {}
    with open(conf.data.label_token_file) as fin:
        for line in fin:
            content = json.loads(line)
            label = content["label"]
            tokens = content["label_token"]
            if label not in label_map:
                continue
            label_tokens[label] = tokens
            label_emb[label] = np.mean([emb.get(token, np.zeros(conf.embedding.dimension))
                                        for token in tokens], axis=0).astype(np.float32)

    vectorizer = TfidfVectorizer(stop_words='english', dtype=np.float32)
    label_tokens_str = [" ".join(label_tokens[id2label[i]]) for i in range(len(id2label))]
    label_tokens_tfidf = vectorizer.fit_transform(label_tokens_str)
    ordered_tokens = vectorizer.get_feature_names()
    label_ordered_tokens_emb = np.array([[emb.get(token, np.zeros(conf.embedding.dimension))
                                          for token in ordered_tokens]
                                         for _ in range(len(id2label))], dtype=np.float32)
    tf_idf_label_emb = np.einsum('ikl,ik->il', label_ordered_tokens_emb, label_tokens_tfidf.todense())

    print("saving pretrained label embedding. ")
    prefix = conf.data.label_map_of_relation_files.split('/')[1].split('_')[0].strip()
    with open("data/{}_label_avg_embedding.txt".format(prefix), mode='w') as fout:
        for label in label_emb:
            fout.write("{} {}\n".format(label, " ".join(label_emb[label].astype(np.str))))
    with open("data/{}_label_tfidf_embedding.txt".format(prefix), mode='w') as fout:
        for i in range(tf_idf_label_emb.shape[0]):
            label = id2label[i]
            fout.write("{} {}\n".format(label, " ".join(tf_idf_label_emb[i].astype(np.str))))

    # get hierarchy graph
    hier_adj = get_hier_adj(parent_child_map, label_map)
    print("Number of edges in hierarchy graph:", hier_adj.nnz)
    print("save hierarchy adj file. ")
    sp.save_npz(conf.data.label_relation_candidate_files["hierarchy"], hier_adj)

    # get cooc & corr adj
    cooc_adj, corr_adj = get_cooc_corr_adj(id2label, conf, cooc_threshold=args.cooc_threshold)

    print("Number of edges in co-occurrence graph:", cooc_adj.nnz)
    print("save co-occurrence adj file. ")
    sp.save_npz(conf.data.label_relation_candidate_files["cooccurrence"], cooc_adj)

    print("Number of edges in correlation graph:", corr_adj.nnz)
    print("save correlation adj file. ")
    sp.save_npz(conf.data.label_relation_candidate_files["correlation"], corr_adj)

    # get simi adj
    if args.tfidf_simi:
        simi_adj = get_simi_adj(tf_idf_label_emb, simi_threshold=args.simi_threshold)
    else:
        simi_adj = get_simi_adj(np.array([label_emb[id2label[i]] for i in range(len(id2label))]),
                                simi_threshold=args.simi_threshold)
    print("Number of edges in similarity graph:", simi_adj.nnz)
    print("save similarity adj file. ")
    sp.save_npz(conf.data.label_relation_candidate_files["similarity"], simi_adj)

    # impose hier-cooc
    hier_cooc_impose_adj = impose_adjs(hier_adj, cooc_adj)
    print("Number of edges in imposed graph of hierarchy and co-occurrence:", hier_cooc_impose_adj.nnz)
    print("save hierarchy and co-occurrence imposed adj file. ")
    sp.save_npz(conf.data.label_relation_candidate_files["hierarchy_cooccurrence_impose"], hier_cooc_impose_adj)

    # impose hier-corr
    hier_corr_impose_adj = impose_adjs(hier_adj, corr_adj)
    print("Number of edges in imposed graph of hierarchy and correlation:", hier_corr_impose_adj.nnz)
    print("save hierarchy and correlation imposed adj file. ")
    sp.save_npz(conf.data.label_relation_candidate_files["hierarchy_correlation_impose"], hier_corr_impose_adj)

    # impose hier-simi
    hier_simi_impose_adj = impose_adjs(hier_adj, simi_adj)
    print("Number of edges in imposed graph of hierarchy and similarity:", hier_simi_impose_adj.nnz)
    print("save hierarchy and similarity imposed adj file. ")
    sp.save_npz(conf.data.label_relation_candidate_files["hierarchy_similarity_impose"], hier_simi_impose_adj)

    # # impose corr-simi
    # corr_simi_impose_adj = impose_adjs(corr_adj, simi_adj)
    # print("Number of edges in imposed graph of correlation and similarity:", corr_simi_impose_adj.nnz)
    # print("save correlation and similarity imposed adj file. ")
    # sp.save_npz(conf.data.label_relation_candidate_files["correlation_similarity_impose"], corr_simi_impose_adj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--cooc_threshold", type=float, default=5e-05)
    parser.add_argument("--simi_threshold", type=float, default=5e-05)
    parser.add_argument('--tfidf_simi', action='store_true')
    args = parser.parse_args()
    conf = Config(config_file=args.config_file)
    main(conf, args)
