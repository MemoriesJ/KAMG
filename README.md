# Multi-label Few/Zero-shot Learning with Knowledge Aggregated from Multiple Label Graphs


## Introduction 

We intorduce the knowledge aggregation machenism to improve the few/zero shot learning performance on MIMIC-II/III datasets as well as the EU legislation dataset
we extend ([Rios, 2018]) work with the implementation from Neural classifier ([])
To use our codes, please make sure that you have read the README file from [Neural classifier](https://github.com/MemoriesJ/NeuralNLP-NeuralClassifier/blob/master/README.md)

## Support text encoders

ZSJLCNN - ACNN + KAMG in paper
ZLWACNN - Rios' (2018) model

other encoders for conventional text classification task are mentioned in [Neural classifier](https://github.com/MemoriesJ/NeuralNLP-NeuralClassifier/blob/master/README.md)

## Dataset

Due to the limit access to MIMIC dataset, you need to follow the instruction on []() to get all dataset
Tokenization and cleaning codes will be provided very soon

Graph data will be uploaded to Google Drive with open access in order to repeat the experimental results. Link will be provides here very soon. The python scripts for generating graph data has been included in the source codes.


## Acknowledgement

Our codes are developed based on [Neural classifier](https://github.com/MemoriesJ/NeuralNLP-NeuralClassifier/blob/master/README.md)
we have modified following files which will be different from originals

* model/rnn.py                           (update)
* model/zlwacnn.py                       (add)
* model/zlwarnn.py                       (add)
* model/zsjlcnn.py                       (add)
* model/zsjlrnn.py                       (add)
* evaluate/classification_evaluate.py    (update)
* dataset/collator                       (update)
* dataset/dataset.py                     (update)
* dataset/classification_dataset.py      (update)
* dataset/generate_graph_data.py         (add)
* dataset/graph_dataset.py               (add)
* util.py                                (add)