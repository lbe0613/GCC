# GCC
Source code of EMNLP 2021 Findings paper "[Glyph Enhanced Chinese Character Pre-Training for Lexical Sememe Prediction](https://aclanthology.org/2021.findings-emnlp.386)".

## Requirements
* `python`: 3.7.6
* `pytorch`: 1.5.0
* `transformers`: 3.5.0



## Pre-Training
Before training, please generate word pre-training samples. The format is one word per line, take [`pretrain/corpus/train.txt`](https://github.com/lbe0613/GCC/blob/main/pretrain/corpus/train.txt) for example. In our work, We adopt [Tencent embedding corpus](https://ai.tencent.com/ailab/nlp/en/embedding.html). We remove non-Chinese characters such as punctuation and pure digits, and finally get 7,291,828 words as our pre-training samples.

Pre-train the model:
```bash
$ cd pretrain
$ python run_pretrain.py
```

The model pre-trained by us can be downloaded from [GCC](https://pan.baidu.com/s/1uCCtnXexX-Tp-BuMxULyOQ) (password: lqc2). 

[`data/GCC_vocab.json`](https://github.com/lbe0613/GCC/blob/main/data/GCC_vocab.json) is the mapping file for `data/han_seq.json` if you want to know how Chinese characters are decomposed.


## Finetune
We use the same Sememe Prediction dataset as [SCorP](https://github.com/thunlp/SCorP/tree/master/data) as well as the word embedding pretrained by them.
```bash
$ cp -r https://github.com/thunlp/SCorP/blob/master/data finetune/data
```

Finetune the model:
```bash
$ cd finetune
$ python run_finetune.py
```
The Sememe Prediction model trained by us can be downloaded from [GCC_SP](https://pan.baidu.com/s/1GGXmqSEU-YAhFsIeetK_IQ) (password: nb0a).

The MAP on test dataset should be about 60.23.
## Cite
If you find our code is useful, please cite:
```
@inproceedings{lyu-etal-2021-glyph-enhanced,
    title = "Glyph Enhanced {C}hinese Character Pre-Training for Lexical Sememe Prediction",
    author = "Lyu, Boer and Chen, Lu and Yu, Kai",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.386",
    pages = "4549--4555",
    abstract = "Sememes are defined as the atomic units to describe the semantic meaning of concepts. Due to the difficulty of manually annotating sememes and the inconsistency of annotations between experts, the lexical sememe prediction task has been proposed. However, previous methods heavily rely on word or character embeddings, and ignore the fine-grained information. In this paper, we propose a novel pre-training method which is designed to better incorporate the internal information of Chinese character. The Glyph enhanced Chinese Character representation (GCC) is used to assist sememe prediction. We experiment and evaluate our model on HowNet, which is a famous sememe knowledge base. The experimental results show that our method outperforms existing non-external information models.",
}
```
