# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 15:02
# @Author  : wanli.li
# @Email   : wanli.li@m.scnu.edu.cn
# @File    : selection.py
# @Software: PyCharm
import math
import torch
import ipdb
import json
from torchtext import data
from utils import scorer

TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
RELATION = data.Field(sequential=False, unk_token=None, pad_token=None)
POS = data.Field(sequential=True, batch_first=True)
NER = data.Field(sequential=True, batch_first=True)
PST = data.Field(sequential=True, batch_first=True)
ID = data.Field(sequential=False, use_vocab=False, dtype=torch.int)
PR_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
SL_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

FIELDS = {
    'tokens': ('token', TOKEN),
    'stanford_pos': ('pos', POS),
    'stanford_ner': ('ner', NER),
    'relation': ('relation', RELATION),
    'subj_pst': ('subj_pst', PST),
    'obj_pst': ('obj_pst', PST),
    'pr_confidence': ('pr_confidence', PR_CONFIDENCE),
    'sl_confidence': ('sl_confidence', SL_CONFIDENCE),
    'id': ('id',ID)
}

def find_example_by_id(examples, eid):
    for example in examples:
        if example.id==eid:
            return example

def find_example_by_id_tac(examples, eid):
    for example in examples:
        if example.id==int(eid):
            return example

def example_to_dict(example, pr_confidence, sl_confidence, rel):
    output = {}
    output['id'] = example.id
    output['tokens'] = example.token
    output['stanford_pos'] = example.pos
    output['stanford_ner'] = example.ner
    output['subj_pst'] = example.subj_pst
    output['obj_pst'] = example.obj_pst
    output['relation'] = rel
    output['pr_confidence'] = pr_confidence
    output['sl_confidence'] = sl_confidence
    return output

def example_to_dict_tac(example, pr_confidence, sl_confidence, rel):
    output = {}
    output['id'] = example.id
    output['tokens'] = example.token
    output['stanford_pos'] = example.pos
    output['stanford_ner'] = example.ner
    output['subj_pst'] = example.subj_pst
    output['obj_pst'] = example.obj_pst
    output['relation'] = rel
    output['pr_confidence'] = pr_confidence
    output['sl_confidence'] = sl_confidence
    return output

def select_samples(meta_idxs_p, confidence_idxs_p, meta_idxs_g, confidence_idxs_g, dataset_infer,k_samples, batch_size, integrate_methods, is_tac=False):
    print('Infer on predictor: ')  # Track performance of predictor alone
    # gold, guess = [t[2] for t in meta_idxs_p], [t[1] for t in meta_idxs_p]
    gold, guess = [t[2] for t in meta_idxs_p][:k_samples], [t[1] for t in meta_idxs_p][:k_samples]
    scorer.score(gold, guess, verbose=False)
    gold, guess = [t[2] for t in meta_idxs_g][:k_samples], [t[1] for t in meta_idxs_g][:k_samples]
    scorer.score(gold, guess, verbose=False)
    upperbound=k_samples
    # Case Study
    # Case = []
    # for x in meta_idxs_g:
    #    if x not in meta_idxs_p:
    #        Case.append(x)
    # with open('dataset/case/case-0.05.json', 'w') as t:
    #     json.dump(Case,t)
    # with open('dataset/case/mp-0.05.json', 'w') as t:
    #     json.dump(meta_idxs_p,t)
    # for intersaction
    if integrate_methods=='intersection':
        meta_idxs=[]
        while len(meta_idxs) < min(k_samples, len(meta_idxs_p)):
            upperbound = math.ceil(1.25 * upperbound)
            meta_idxs = sorted(set(meta_idxs_g).intersection(
                set(meta_idxs_p)))[:k_samples]
            if upperbound > k_samples * 10:  # set a limit for growing upperbound
                break
        print('intersection num:',len(meta_idxs))
        for x in meta_idxs_p:
            if x not in meta_idxs and len(meta_idxs)<k_samples:
                meta_idxs.append(x)
        print('new examples:',len(meta_idxs))

    if integrate_methods == 'p_only':
        meta_idxs = meta_idxs_p[:k_samples]
    if integrate_methods == 'g_only':
        meta_idxs = meta_idxs_g[:k_samples]
        if len(meta_idxs)<k_samples:
            for i in meta_idxs_p:
                if i not in meta_idxs and len(meta_idxs) < k_samples:
                    meta_idxs.append(i)
    iterator_unlabeled = data.Iterator(
        dataset=dataset_infer,
        batch_size=batch_size,
        device=-1,
        repeat=False,
        train=False,
        shuffle=False,
        sort=True,
        sort_key=lambda x: -len(x.token),
        sort_within_batch=False)
    examples = iterator_unlabeled.data()
    new_examples, rest_examples, example_ids = [], [], set(idx for idx, pred, actual in meta_idxs)
    meta_idxs = [(idx, pred, actual, 1.0, 1.0) for idx, pred, actual in meta_idxs]

    for idx, pred, _, pr_confidence, sl_confidence in meta_idxs:
        if is_tac:
            output = example_to_dict_tac(find_example_by_id_tac(examples, str(idx)), pr_confidence, sl_confidence, pred)
        else:
            output = example_to_dict(find_example_by_id(examples, str(idx)), pr_confidence, sl_confidence, pred)
        new_examples.append(data.Example.fromdict(output, FIELDS))
    rest_examples = [example for example in examples if int(example.id) not in example_ids]
    print('New examples num :', len(meta_idxs))
    gold, guess = [t[2] for t in meta_idxs], [t[1] for t in meta_idxs]
    scorer.score(gold, guess, verbose=False)
    return new_examples, rest_examples, integrate_methods