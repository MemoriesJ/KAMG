# KAMG

Code repository for EMNLP 2020 proceedings paper [Multi-label Few/Zero-shot Learning with Knowledge Aggregated from Multiple Label Graphs](https://arxiv.org/abs/2010.07459).


## Introduction 

We intorduce the knowledge aggregation machenism to improve the few/zero shot learning performance on MIMIC-II/III datasets as well as the EU legislation dataset。
we extend [Rios & Kavuluru, 2018](https://www.aclweb.org/anthology/D18-1352/) work with the implementation from Neural classifier ([Liu, Mu, Li, Mu, Tang, Ai, ... & Zhou, 2019](https://github.com/Tencent/NeuralNLP-NeuralClassifier))
To use our codes, please make sure that you have read the README file from [Neural classifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier)

## Support text encoders

* ZSJLCNN - [ACNN + KAMG in our paper](https://arxiv.org/abs/2010.07459)
* ZLWACNN - [Rios & Kavuluru (2018) model](https://www.aclweb.org/anthology/D18-1352/)

other encoders for conventional text classification task are mentioned in [Neural classifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier)

## Dataset

Due to the limit access to MIMIC dataset, you need to follow the instruction on the [web page](https://mimic.physionet.org/gettingstarted/access/) to apply for all datasets
Tokenization and cleaning codes will be provided very soon

EU legislation dataset could be download from 
but the codes for running the experiments of EU legislation dataset are developed from [Chalkidis, Fergadiotis, Malakasiotis, & Androutsopoulos, 2019](https://github.com/iliaschalkidis/lmtc-eurlex57k) original codes, so not included here.

Graph data will be uploaded to Google Drive with open access in order to repeat the experimental results. Link will be provides here very soon. The python scripts for generating graph data has been included in the source codes.

## Config files

For quickly repeat our experiments results, the config files will also be 

To simply run the codes, you should 

```
cd codes
python train.py conf/config.py
```

which is same as [Neural classifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier)

## Acknowledgement

Our codes are based on [Neural classifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier)
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

# Citation
```
@inproceedings{lu2020kamg,
  title={Multi-label Few/Zero-shot Learning with Knowledge Aggregated from Multiple Label Graphs},
  author={},
  booktitle={Empirical Methods in Natural Language Processing},
  year={2020}
}
```
