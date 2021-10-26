# -*- coding: utf-8 -*-
# @Time    : 2019/12/3 14:41
# @Author  : wanli.li
# @Email   : wanli.li@m.scnu.edu.cn
# @File    : train.py
# @Software: PyCharm
import sys
import argparse
import torch
import random
import math
import os
from model.predictor import Predictor
from model.trainer import Trainer,evaluate,encode
from selection import select_samples
from torch.autograd import Variable
from model.inferencer_sp import Inferencer
import torch.optim as optim
from utils import scorer
from torch import nn
import torch.nn.functional as F
import copy
import logging
import time
import numpy as np
from utils.torch_utils import batch_to_input

from torchtext import data
from utils import helper,torch_utils
import ipdb

os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()

# logger = logging.getLogger()
# logger.setLevel(logging.ERROR)  # Log等级总开关
# # 第二步，创建一个handler，用于写入日志文件
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '/TASLP/Logs/'
# log_name = log_path + rq + '.log'
# logfile = log_name
# fh = logging.FileHandler(logfile, mode='w')
# fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# # 第三步，定义handler的输出格式
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# fh.setFormatter(formatter)
# # 第四步，将logger添加到handler里面
# logger.addHandler(fh)

# Begin Encoder specific arguments
parser.add_argument(
    '--encoder_method',
    type=str,
    default='PRNN',
    choices=['PRNN'],
    help='Method to encode sentences.')
parser.add_argument(
    '--num_iters',
    type=int,
    default=-1,
    help='# of iterations. -1 indicates it\'s determined by data_ratio.')
parser.add_argument(
    '--alpha', type=float, default=0.5, help='confidence hyperparameter for encoder.')
parser.add_argument(
    '--beta', type=float, default=0.5, help='confidence hyperparameter for GCN.')
parser.add_argument(
    '--continue_training',
    type=bool,
    default=False,
    help=
    'whether to start over again for self-training/dualre/re-ensemble methods, default is start over.'
)
parser.add_argument(
    '--bidirectional', default=True, type=bool, help='whether to use bidirectional RNN.')

# Begin original dataset arguments
parser.add_argument('--predictor_dir', type=str, default='saved_models/encoder', help='Directory of the encoder.')
parser.add_argument('--gat_dir', type=str, default='saved_models/gat/', help='Directory of the GCN.')
parser.add_argument('--data_dir', type=str, default='dataset/semeval')
parser.add_argument('--labeled_ratio', type=float, default=0.1)
parser.add_argument('--unlabeled_ratio', type=float, default=0.5)
# ratio of instances to promote each round
parser.add_argument('--data_ratio', type=float, default=0.1)
parser.add_argument('--FEATYPE', type=str, default='fea')

parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--p_dropout', type=float, default=0.5, help='Input and RNN dropout rate.')


parser.add_argument('--gat_input_dim', type=int, default=219, help='GAT input feature size.')
parser.add_argument('--gat_hid_dim', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--gat_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument(
    '--gat_dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--gat_alpha', type=float, default=0.5, help='Alpha for the leaky_relu.')
parser.add_argument('--gat_decay', type=float, default=5e-4, help='Weight decay for the optimizer.')


parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
parser.add_argument('--no-attn', dest='attn', action='store_false')
parser.set_defaults(attn=True)
parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')


parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--gat_lr', type=float, default=0.005, help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=60)
parser.add_argument('--gat_epoch', type=int, default=60, help='Epoch of the GCN.')
parser.add_argument('--patience', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument(
    '--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument(
    '--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--feature_probs', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)
# torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

args, _ = parser.parse_known_args()
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
# is_gpu = torch.cuda.is_available()
# gpu_nums = torch.cuda.device_count()
# gpu_index = torch.cuda.current_device()
# print(is_gpu,gpu_nums,gpu_index)
# make opt
opt = vars(args)

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
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

dataset_vocab = data.TabularDataset(
    path=opt['data_dir'] + '/train' + '.json',
    format='json',
    fields=FIELDS)
dataset_train = data.TabularDataset(
    path=opt['data_dir'] + '/train-' + str(opt['labeled_ratio']) + '.json',
    format='json',
    fields=FIELDS)
dataset_link = data.TabularDataset(
    path=opt['data_dir'] + '/train-' + str(opt['labeled_ratio']) + '.json',
    format='json',
    fields=FIELDS)
dataset_infer = data.TabularDataset(
    path=opt['data_dir'] + '/raw-' + str(opt['unlabeled_ratio']) + '.json',
    format='json',
    fields=FIELDS)
dataset_dev = data.TabularDataset(path=opt['data_dir'] + '/dev.json', format='json', fields=FIELDS)
dataset_test = data.TabularDataset(
    path=opt['data_dir'] + '/test.json', format='json', fields=FIELDS)

print('=' * 100)
print('Labeled data path: ' + opt['data_dir'] + '/train-' + str(opt['labeled_ratio']) + '.json')
print('Unlabeled data path: ' + opt['data_dir'] + '/raw-' + str(opt['unlabeled_ratio']) + '.json')
print('Labeled instances #: %d, Unlabeled instances #: %d' % (len(dataset_train.examples),
                                                              len(dataset_infer.examples)))
print('=' * 100)

TOKEN.build_vocab(dataset_vocab)
RELATION.build_vocab(dataset_vocab)
POS.build_vocab(dataset_vocab)
NER.build_vocab(dataset_vocab)
PST.build_vocab(dataset_vocab)

opt['num_class'] = len(RELATION.vocab)
opt['vocab_pad_id'] = TOKEN.vocab.stoi['<pad>']
opt['pos_pad_id'] = POS.vocab.stoi['<pad>']
opt['ner_pad_id'] = NER.vocab.stoi['<pad>']
opt['pe_pad_id'] = PST.vocab.stoi['<pad>']
opt['vocab_size'] = len(TOKEN.vocab)
opt['pos_size'] = len(POS.vocab)
opt['ner_size'] = len(NER.vocab)
opt['pe_size'] = len(PST.vocab)
opt['rel_stoi'] = RELATION.vocab.stoi
opt['rel_itos'] = RELATION.vocab.itos

helper.ensure_dir(opt['predictor_dir'], verbose=True)
helper.ensure_dir(opt['gat_dir'], verbose=True)

# TOKEN.vocab.load_vectors('glove.840B.300d', cache='./dataset/.vectors_cache')
TOKEN.vocab.load_vectors('glove.840B.300d', cache='../TASLP/dataset/.vectors_cache')
# TOKEN.vocab.load_vectors(
#     'glove.840B.300d',
#     cache='./dataset/.vectors_cache',
#     unk_init=functools.partial(torch.nn.init.uniform_, a=-1, b=1))  # randomly
if TOKEN.vocab.vectors is not None:
    opt['emb_dim'] = TOKEN.vocab.vectors.size(1)

num_iters = math.ceil(1.0 / opt['data_ratio'])
if args.num_iters > 0:
    num_iters = min(num_iters, args.num_iters)
k_samples = math.ceil(len(dataset_infer.examples) * opt['data_ratio'])
dev_f1_iter, dev_pr_iter,dev_re_iter, test_f1_iter, test_pr_iter, test_re_iter= [], [], [], [], [], []
predictor, gcn = None, None

def load_best_model(model_dir, model_type='predictor'):
    model_file = model_dir + '/best_model.pt'
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    predictor = Predictor(model_opt)
    model = Trainer(model_opt, predictor, model_type=model_type)
    model.load(model_file)
    helper.print_config(model_opt)
    return model

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def get_link_edge(coo, linked_infer_id):
    coo = coo[linked_infer_id]
    coo = coo[:,linked_infer_id]
    tmp = coo - np.eye(coo.shape[0])
    x, y = np.nonzero(tmp)
    return x,y

def prune_graph(graph, id):
    graph = graph[id]
    graph = graph[:, id]
    return torch.tensor(graph)

def construct_linked_dataset(dataset_train,dataset_infer,dataset_link,linked_set):
    linked_examples =[]
    for example in dataset_infer.examples:
        if int(example.id) in linked_set:
            linked_examples.append(example)
    for example in dataset_train.examples:
        if int(example.id) in linked_set:
            linked_examples.append(example)
    dataset_link.examples = linked_examples
    return dataset_link

def construct_all_dataset(dataset_train,dataset_infer,dataset_link):
    linked_examples =[]
    for example in dataset_train.examples:
            linked_examples.append(example)
    for example in dataset_infer.examples:
            linked_examples.append(example)
    dataset_link.examples = linked_examples
    return dataset_link

def change_adj(x,y,pred_adj,coo,golds,linked_infer_id,threshold):
    right_num = 0
    error_num = 0
    for x_val, y_val in zip(x,y):
        if pred_adj[x_val][y_val] < threshold:
            coo[linked_infer_id[x_val]][linked_infer_id[y_val]] = 0
            if golds[x_val] == golds[y_val]:
                error_num += 1
            else:
                right_num += 1
    print('Right num is {}. Right Rate is {}'.format(right_num,right_num/(right_num+error_num+1)))

def data_to_input(batch, id_train, vocab_pad_id=0):
    inputs = {}
    num_train,num_other = [],[]
    inputs['id'] = batch.id
    inputs['words'], inputs['length'] = batch.token
    inputs['pos'] = batch.pos
    inputs['ner'] = batch.ner
    inputs['subj_pst'] = batch.subj_pst
    inputs['obj_pst'] = batch.obj_pst
    inputs['masks'] = torch.eq(batch.token[0], vocab_pad_id)
    inputs['pr_confidence'] = batch.pr_confidence
    inputs['sl_confidence'] = batch.sl_confidence
    for i,id in enumerate(batch.id):
        if id in id_train:
            num_train.append(i)
        else:
            num_other.append(i)
    return inputs, batch.relation, num_train, num_other

def get_id(examples):
    id = []
    for example in examples:
        id.append(int(example.id))
    return id

def train_inferencer(opt,inferencer,optimizer,adj,features,target,id_train,id_test):
    t = time.time()
    # criterion = nn.CrossEntropyLoss(reduction='none')

    if opt['cuda']:
        inferencer.cuda()
        adj = adj.cuda()
    for epoch in range(opt['gat_epoch']):
        inferencer.train()
        optimizer.zero_grad()
        logits, _ = inferencer(features,adj)
        loss = F.nll_loss(logits[id_train], target[id_train])
        loss = torch.mean(loss)
        acc_train = accuracy(logits[id_train], target[id_train])
        loss.backward()
        optimizer.step()
        inferencer.eval()
        logits, _ = inferencer(features, adj)
        loss_val = F.nll_loss(logits[id_test],target[id_test])
        loss_val = torch.mean(loss_val)
        acc_val = accuracy(logits[id_test], target[id_test])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))
    inferencer.eval()
    # retrieve
    confidences = []
    preds = inferencer.predict(features, adj).cpu().detach().numpy()[id_test]
    pred_label = np.argmax(preds, 1).tolist()
    for j in preds:
        confidences.append(max(j) ** opt['beta'])
    pred_rel = list(map(lambda x: opt['rel_itos'][x], pred_label))
    gold_rel = list(map(lambda x: opt['rel_itos'][x], golds[id_test]))
    scorer.score(gold_rel, pred_rel, verbose=False)
    sample_id = []
    for i in id_test:
        sample_id.append(all_id[i])
    ranking = list(zip(sample_id, pred_rel, gold_rel, confidences))
    ranking = sorted(
        ranking, key=lambda x: x[3], reverse=True)
    meta_idxs_g, confidence_idxs_g = [], []
    for id, rel, gold, confidence in ranking:
        meta_idxs_g.append((id, rel, gold))
        confidence_idxs_g.append((id, confidence))
    return meta_idxs_g, confidence_idxs_g

integrate_methods = 'intersection'
start = time.process_time()
for num_iter in range(num_iters + 1):
    print('')
    print('=' * 100)
    print(
        'Training #: %d, Infer #: %d' % (len(dataset_train.examples), len(dataset_infer.examples)))

    train_id = get_id(dataset_train.examples)
    # ====================== #
    # Begin Train on Predictor
    # ====================== #
    print('Training on iteration #%d for Predictor...' % num_iter)
    opt['model_save_dir'] = opt['predictor_dir']
    opt['dropout'] = opt['p_dropout']

    # save config
    helper.save_config(opt, opt['model_save_dir'] + '/config.json', verbose=False)
    helper.print_config(opt)

    # prediction module
    if predictor is None or not opt['continue_training']:
        predictor = Predictor(opt, emb_matrix=TOKEN.vocab.vectors)
        model = Trainer(opt, predictor, model_type='predictor')
    model.train(dataset_train, dataset_dev)
    #
    # Evaluate
    best_model_p = load_best_model(opt['model_save_dir'], model_type='predictor')
    print('Final evaluation #%d on train set...' % num_iter)
    evaluate(best_model_p, dataset_train, verbose=False)
    print('Final evaluation #%d on dev set...' % num_iter)
    best_model_p = load_best_model(opt['model_save_dir'], model_type='predictor')
    print('Final evaluation #%d on train set...' % num_iter)
    evaluate(best_model_p, dataset_train, verbose=False)
    print('Final evaluation #%d on dev set...' % num_iter)
    dev_p, dev_r, dev_f1 = evaluate(best_model_p, dataset_dev, verbose=False)[:3]
    print('Final evaluation #%d on test set...' % num_iter)
    test_p, test_r, test_f1 = evaluate(best_model_p, dataset_test, verbose=False)[:3]
    dev_f1_iter.append(dev_f1)
    dev_pr_iter.append(dev_p)
    dev_re_iter.append(dev_r)
    test_f1_iter.append(test_f1)
    test_pr_iter.append(test_p)
    test_re_iter.append(test_r)

    best_model_p = load_best_model(opt['predictor_dir'], model_type='predictor')

    meta_idxs_p, confidence_idxs_p = best_model_p.retrieve(dataset_infer,len(dataset_infer))

    # ====================== #
    # Begin Train on Inferencer
    # ====================== #
    if num_iter==10:
        preds,results = [],[]
        iterator_test = data.Iterator(
            dataset=dataset_test,
            batch_size=model.opt['batch_size'],
            device=None,
            repeat=False,
            train=True,
            shuffle=False,
            sort=True,
            sort_key=lambda x: -len(x.token),
            sort_within_batch=False)
        for batch in iterator_test:
            inputs, target = batch_to_input(batch, model.opt['vocab_pad_id'])
            prediction = best_model_p.predict(inputs)
            results.extend([prediction[1][i][p] for i,p in enumerate(prediction[0])])
            preds.extend(prediction)
        results = np.array(results)
        np.save('one_results_{}.npy'.format(str(opt['labeled_ratio'])),results)
        break
    print('Training on iteration #%d for Inferencer...' % num_iter)
    opt['model_save_dir'] = opt['gat_dir']
    opt['dropout'] = opt['gat_dropout']
    # all_coo, pred_coo, en_coo = np.load('dataset/graph/item_coo-%s.npy'%opt['labeled_ratio'], allow_pickle=True)
    all_coo, pred_coo, en_coo = np.load('dataset/graph_se/item_coo.npy', allow_pickle=True)

    [x, y] = all_coo[train_id].nonzero()
    linked_set = set(y)
    dataset_link = construct_linked_dataset(dataset_train, dataset_infer, dataset_link, linked_set)
    features, probs, all_id, golds, idx_train, idx_test = encode(best_model_p, dataset_link, train_id)

    if opt['feature_probs']:
        features = torch.cat([features, probs], dim=1).detach()
    adj = torch.FloatTensor(all_coo[all_id][:, all_id])
    # tmp_coo = all_coo[all_id][:, all_id]
    # [x, y] = tmp_coo.nonzero()
    # indices = torch.tensor([x, y])
    # values = torch.ones(len(x))
    # adj = torch.sparse_coo_tensor(indices, values, tmp_coo.shape)
    # save config

    inferencer = Inferencer(opt,emb_matrix=TOKEN.vocab.vectors)

    optimizer = optim.Adam(inferencer.parameters(),
                           lr=opt['gat_lr'],
                           weight_decay=opt['gat_decay'])
    meta_idxs_g, confidence_idxs_g = train_inferencer(opt,inferencer,optimizer,adj,features,golds,idx_train,idx_test)
    new_examples, rest_examples, integrate_methods = select_samples(meta_idxs_p, confidence_idxs_p,
                                                                    meta_idxs_g, confidence_idxs_g,
                                                                    dataset_infer,
                                                                    k_samples, 50, integrate_methods)
    dataset_train.examples = dataset_train.examples + new_examples
    dataset_infer.examples = rest_examples
    torch.cuda.empty_cache()

# update dataset
scorer.print_table(
            dev_pr_iter, dev_re_iter,dev_f1_iter, test_pr_iter, test_re_iter, test_f1_iter, header='Best dev and test F1 with seed=%s:' % args.seed)
end = time.process_time()

n_trainable_params, n_nontrainable_params = 0, 0
for p in predictor.parameters():
    n_params = torch.prod(torch.tensor(p.shape))
    if p.requires_grad:
        n_trainable_params += n_params
    else:
        n_nontrainable_params += n_params
print(
    'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
for p in inferencer.parameters():
    n_params = torch.prod(torch.tensor(p.shape))
    if p.requires_grad:
        n_trainable_params += n_params
    else:
        n_nontrainable_params += n_params
print(
    'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
print('Running time: %s Seconds' % (end - start))