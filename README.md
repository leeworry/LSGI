# MRefG
Wanli Li and Tieyun Qian: "[Exploit a Multi-head Reference Graph for Semi-supervised Relation Extraction](https://ieeexplore.ieee.org/document/9534434)", IJCNN 2021
## 1. Requirements
 To reproduce the reported results accurately, please install the specific version of each package.
* python 3.7.10
* torch 1.7.1
* numpy 1.19.2

All data should be put into `dataset/$data_name` folder in a similar format as `dataset/sample`, with a naming convention such that (1) `train-$ratio.json` indicates that certain percentage of training data are used. (2) `raw-$ratio.json` is a part of original training data, in which we assume the labels are unknown to model.

To replicate the experiments, first prepare the required dataset as below:

- SemEval: SemEval 2010 Task 8 data (included in `dataset/semeval`)
- TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))
  - Put the official dataset (in JSON format) under folder `dataset/tacred` in a similar format like [here](https://github.com/yuhaozhang/tacred-relation/tree/master/dataset/tacred).

Then use the scripts from `utils/data_utils.py` to further preprocess the data. For SemEval, the script split the original training data into two sets (labeled and unlabeled) and then separate them into multiple ratios. For TACRED, the script first perform some preprocessing to ensure the same format as SemEval.

We provide our data for reproducing the reported results.

## Code Overview
The main entry for all models is in `train_sp.py`. We provide the sparse graph model.

## Citation

```latex
@inproceedings{DBLP:conf/ijcnn/LiQCTZZ21,
  author    = {Wanli Li and
               Tieyun Qian and
               Xu Chen and
               Kejian Tang and
               Shaohui Zhan and
               Tao Zhan},
  title     = {Exploit a Multi-head Reference Graph for Semi-supervised Relation
               Extraction},
  booktitle = {International Joint Conference on Neural Networks, {IJCNN} 2021, Shenzhen,
               China, July 18-22, 2021},
  pages     = {1--7},
  publisher = {{IEEE}},
  year      = {2021},
}
```
