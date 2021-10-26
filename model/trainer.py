# -*- coding: utf-8 -*-
# @Time    : 2020/2/4 15:18
# @Author  : wanli.li
# @Email   : wanli.li@m.scnu.edu.cn
# @File    : trainer.py
# @Software: PyCharm
"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import time
import os
from datetime import datetime
from shutil import copyfile
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data
from sklearn.preprocessing import normalize

from utils import torch_utils, scorer
from utils.torch_utils import batch_to_input, arg_max

def kl_div_with_logit(q_logit, p_logit):
    logq = F.softmax(q_logit, dim=1)
    logp = F.softmax(p_logit, dim=1)
    loss = torch.sum(F.kl_div(logq, logp))
    return loss


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2)) + 1e-16)
    return torch.from_numpy(d)

def idx_to_onehot(target, opt, confidence=None):
    sample_size, class_size = target.size(0), opt['num_class']
    if confidence is None:
        y = torch.zeros(sample_size, class_size)
        y = y.scatter_(1, torch.unsqueeze(target.data, dim=1), 1)
    else:
        y = torch.ones(sample_size, class_size)
        y = y * (1 - confidence.data).unsqueeze(1).expand(-1, class_size)
        y[torch.arange(sample_size).long(), target.data] = confidence.data

    y = Variable(y)

    return y

def encode(model, dataset, train_id, cuda=True):
    rel_stoi, rel_itos = model.opt['rel_stoi'], model.opt['rel_itos']
    iterator_test = data.Iterator(
        dataset=dataset,
        batch_size=model.opt['batch_size'],
        device=None,
        repeat=False,
        train=True,
        shuffle=False,
        sort=True,
        sort_key=lambda x: -len(x.token),
        sort_within_batch=False)
    ids,golds,probs = [],[],[]
    encode = None

    for batch in iterator_test:
        ids.extend(batch.id.tolist())
        golds.extend(batch.relation.tolist())
        inputs, target = batch_to_input(batch, model.opt['vocab_pad_id'])
        if encode is None:
            predictions, probs, loss, encode = model.predict(inputs, target, encoding=True)
        else:
            predictions, tmp_probs, loss, tmp_code = model.predict(inputs, target, encoding=True)
            encode = torch.cat([encode, tmp_code], dim=0)
            probs.extend(tmp_probs)
    idx_train,idx_test = [],[]
    for id,i in enumerate(ids):
        if i in train_id:
            idx_train.append(id)
        else:
            idx_test.append(id)
    if cuda:
        return encode,torch.FloatTensor(probs).cuda(),ids,torch.tensor(golds).cuda(),idx_train,idx_test
    else:
        return encode,torch.FloatTensor(probs),ids,torch.tensor(golds),idx_train,idx_test

def evaluate(model, dataset, evaluate_type='prf', verbose=False):
    rel_stoi, rel_itos = model.opt['rel_stoi'], model.opt['rel_itos']
    iterator_test = data.Iterator(
        dataset=dataset,
        batch_size=model.opt['batch_size'],
        device=None,
        repeat=False,
        train=True,
        shuffle=False,
        sort=True,
        sort_key=lambda x: -len(x.token),
        sort_within_batch=False)

    if evaluate_type == 'prf':
        predictions = []
        all_probs = []
        golds = []
        all_loss = 0
        for batch in iterator_test:
            inputs, target = batch_to_input(batch, model.opt['vocab_pad_id'])
            preds, probs, loss = model.predict(inputs, target)
            predictions += preds
            all_probs += probs
            all_loss += loss
            golds += target.data.tolist()
        predictions = [rel_itos[p] for p in predictions]
        golds = [rel_itos[p] for p in golds]
        p, r, f1 = scorer.score(golds, predictions, verbose=verbose)
        return p, r, f1, all_loss
    elif evaluate_type == 'auc':
        logits, labels = [], []
        for batch in iterator_test:
            inputs, target = batch_to_input(batch, model.opt['vocab_pad_id'])
            logits += model.predict(inputs)[0]
            labels += batch.relation.data.numpy().tolist()
        p, q = 0, 0
        for rel in range(len(rel_itos)):
            if rel == rel_stoi['no_relation']:
                continue
            logits_rel = [logit[rel] for logit in logits]
            labels_rel = [1 if label == rel else 0 for label in labels]
            ranking = list(zip(logits_rel, labels_rel))
            ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
            logits_rel, labels_rel = zip(*ranking)
            p += scorer.AUC(logits_rel, labels_rel)
            q += 1

        dev_auc = p / q * 100
        return dev_auc, None, None, None


def calc_confidence(probs, exp):
    '''Calculate confidence score from raw probabilities'''
    return max(probs)**exp

def find_example_by_id(examples, eid):
    for example in examples:
        if example.id==eid:
            return example


class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, model, model_type='predictor'):
        self.opt = opt
        self.model_type = model_type
        self.model = model
        if model_type == 'predictor':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def train(self, dataset_train, dataset_dev):
        opt = self.opt.copy()
        iterator_train = data.Iterator(
            dataset=dataset_train,
            batch_size=opt['batch_size'],
            device=None,
            repeat=False,
            train=True,
            shuffle=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        iterator_dev = data.Iterator(
            dataset=dataset_dev,
            batch_size=opt['batch_size'],
            device=None,
            repeat=False,
            train=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        dev_score_history = []
        current_lr = opt['lr']

        global_step = 0
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
        max_steps = len(iterator_train) * opt['num_epoch']

        # start training
        epoch = 0
        patience = 0
        while True:
            epoch = epoch + 1
            train_loss = 0

            for batch in iterator_train:
                start_time = time.time()
                global_step += 1
                inputs, target = batch_to_input(batch, opt['vocab_pad_id'])
                loss = self.update(inputs, target)
                train_loss += loss
                if global_step % opt['log_step'] == 0:
                    duration = time.time() - start_time
                    print(
                        format_str.format(datetime.now(), global_step, max_steps, epoch,
                                          opt['num_epoch'], loss, duration, current_lr))

            # eval on dev
            print("Evaluating on dev set...")
            dev_p, dev_r, dev_score, dev_loss = evaluate(self, dataset_dev)

            # print training information
            train_loss = train_loss / len(iterator_train) # avg loss per batch
            dev_loss = dev_loss / len(iterator_dev)
            print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_score = {:.4f}".format(
                epoch, train_loss, dev_loss, dev_score))

            # save the current model
            model_file = opt['model_save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
            self.save(model_file, epoch)
            if epoch == 1 or dev_score > max(dev_score_history):  # new best
                copyfile(model_file, opt['model_save_dir'] + '/best_model.pt')
                print("new best model saved.")
                patience = 0
            else:
                patience = patience + 1
            if epoch % opt['save_epoch'] != 0:
                os.remove(model_file)

            # change learning rate
            if len(dev_score_history) > 10 and dev_score <= dev_score_history[-1] and \
                    opt['optim'] in ['sgd', 'adagrad']:
                current_lr *= opt['lr_decay']
                self.update_lr(current_lr)

            dev_score_history += [dev_score]
            print("")
            if opt['patience'] != 0:
                if patience == opt['patience'] and epoch > opt['num_epoch']:
                    break
            else:
                if epoch == opt['num_epoch']:
                    break
        print("Training ended with {} epochs.".format(epoch))

    def get_vat_loss(self, inputs, target, input_size=360, xi=1e-6, eps=2.5):
        d = torch.Tensor(input_size).normal_()
        d = xi * _l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)

        output_logits, encoding = self.model(inputs)

        logits, encoding = self.model(inputs, direction=d)
        delta_kl = 0.01 * kl_div_with_logit(output_logits.detach(), logits)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        self.model.zero_grad()
        d = _l2_normalize(d)
        d = Variable(d.cuda())
        r_adv = eps * d
        # compute lds
        y_hat, _ = self.model(inputs, r_adv.detach())
        loss = self.criterion(y_hat, target)
        loss = 0.05 * torch.mean(loss)
        # delta_kl = kl_div_with_logit(output_logits.detach(), y_hat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.item()
        return loss_val

    def predict_vat_loss(self, inputs, target=None):
        if self.opt['cuda']:
            target = target.cuda()
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])
        self.model.train()
        self.optimizer.zero_grad()
        vat_loss = self.get_vat_loss(inputs, target)
        return vat_loss

    def train_vat(self, dataset_train, dataset_dev):
        opt = self.opt.copy()
        iterator_train = data.Iterator(
            dataset=dataset_train,
            batch_size=opt['batch_size'],
            repeat=False,
            train=True,
            shuffle=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        iterator_dev = data.Iterator(
            dataset=dataset_dev,
            batch_size=opt['batch_size'],
            repeat=False,
            train=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        dev_score_history = []
        current_lr = opt['lr']

        global_step = 0
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
        max_steps = len(iterator_train) * opt['num_epoch']

        # start training
        epoch = 0
        patience = 0
        while True:
            epoch = epoch + 1
            train_loss = 0

            for batch in iterator_train:
                start_time = time.time()
                global_step += 1

                inputs, target = batch_to_input(batch, opt['vocab_pad_id'])
                loss = self.update(inputs, target)
                vat_loss = self.predict_vat_loss(inputs, target)
                train_loss = loss + 0.1 * vat_loss
                if global_step % opt['log_step'] == 0:
                    duration = time.time() - start_time
                    print(
                        format_str.format(datetime.now(), global_step, max_steps, epoch,
                                          opt['num_epoch'], loss, duration, current_lr))

            # eval on dev
            print("Evaluating on dev set...")
            if self.model_type == 'predictor':
                dev_p, dev_r, dev_score, dev_loss = evaluate(self, dataset_dev)
            else:
                dev_score = evaluate(self, dataset_dev, evaluate_type='auc')[0]
                dev_loss = dev_score

            # print training information
            train_loss = train_loss / len(iterator_train) * opt['batch_size']  # avg loss per batch
            dev_loss = dev_loss / len(iterator_dev) * opt['batch_size']
            print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_score = {:.4f}".format(
                epoch, train_loss, dev_loss, dev_score))

            # save the current model
            model_file = opt['model_save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
            self.save(model_file, epoch)
            if epoch == 1 or dev_score > max(dev_score_history):  # new best
                copyfile(model_file, opt['model_save_dir'] + '/best_model.pt')
                print("new best model saved.")
                patience = 0
            else:
                patience = patience + 1
            if epoch % opt['save_epoch'] != 0:
                os.remove(model_file)

            # change learning rate
            if len(dev_score_history) > 10 and dev_score <= dev_score_history[-1] and \
                    opt['optim'] in ['sgd', 'adagrad']:
                current_lr *= opt['lr_decay']
                self.update_lr(current_lr)

            dev_score_history += [dev_score]
            print("")
            if opt['patience'] != 0:
                if patience == opt['patience'] and epoch > opt['num_epoch']:
                    break
            else:
                if epoch == opt['num_epoch']:
                    break
        print("Training ended with {} epochs.".format(epoch))

    def retrieve(self, dataset, k_samples):

        iterator_unlabeled = data.Iterator(
            dataset=dataset,
            batch_size=self.opt['batch_size'],
            device=None,
            repeat=False,
            train=False,
            shuffle=False,
            sort=True,
            sort_key=lambda x: -len(x.token),
            sort_within_batch=False)

        preds,id = [],[]
        for batch in iterator_unlabeled:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            # print(inputs)
            id.extend(batch.id.tolist())
            preds += self.predict(inputs)[1]
        meta_idxs = []
        confidence_idxs = []
        examples = iterator_unlabeled.data()
        num_instance = len(examples)
        # ranking
        ranking = list(zip(id,range(num_instance), preds))
        ranking = sorted(
            ranking, key=lambda x: calc_confidence(x[2], self.opt['alpha']), reverse=True)
        # selection
        for tid,eid, pred in ranking:
            if len(meta_idxs) == k_samples:
                break
            rid, _ = arg_max(pred)
            val = calc_confidence(pred, self.opt['alpha'])
            rel = self.opt['rel_itos'][rid]
            meta_idxs.append((tid, rel, examples[eid].relation))
            confidence_idxs.append((tid, val))
        return meta_idxs, confidence_idxs

    def construct_graph(self, dataset_train, dataset_infer, threshold=0.9):
        iterator_train = data.Iterator(
            dataset=dataset_train,
            batch_size=len(dataset_train),
            device=None,
            repeat=False,
            train=True,
            shuffle=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        iterator_unlabeled = data.Iterator(
            dataset=dataset_infer,
            batch_size=len(dataset_infer),
            device=None,
            repeat=False,
            train=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)

        for batch in iterator_train:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            # print(inputs)
            train_id = batch.id
            train_rel = batch.relation
            train_num = len(train_id)
            # train_id.extend(batch.id.tolist())
            # train_rel.extend(batch.relation)
            train_preds = self.predict(inputs, target=None, encoding=True)[3]
        # train_no = []
        # for i,rel in enumerate(train_rel):
        #     if self.opt['rel_itos'][rel]!='no_relation':
        #         train_no.append(i)
        # train_num = len(train_no)
        # train_rel = train_rel[train_no]
        # train_preds = train_preds[train_no]
        # train_id = train_id[train_no]

        for batch in iterator_unlabeled:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            # print(inputs)
            infer_num=len(batch.id)
            train_id= torch.cat([train_id,batch.id],dim=0)
            unlabeled_preds = self.predict(inputs, target=None, encoding=True)[3]
            train_rel = torch.cat([train_rel,batch.relation],dim=0)


        all_preds = torch.cat([train_preds,unlabeled_preds], dim=0)
        feature = F.normalize(all_preds)
        adj = feature.mm(feature.t())

        x, y = torch.where(adj[:train_num]>= threshold)

        semantic_coo = torch.eye(train_num+infer_num)
        for i, j in zip(x,y):
            semantic_coo[i,j]=1
        tmp_list = list(range(train_num))
        tmp_list.extend(y.tolist())
        linked_set = list(set(tmp_list))
        semantic_coo = semantic_coo[linked_set][:,linked_set]
        all_preds = all_preds[linked_set]
        train_rel = train_rel[linked_set]
        train_id = train_id[linked_set]

        return semantic_coo,all_preds,train_rel,train_id,train_num,len(linked_set)-train_num

    def construct_graph_bi_tf(self, dataset_train, dataset_infer, threshold=0.9):
        iterator_train = data.Iterator(
            dataset=dataset_train,
            batch_size=len(dataset_train),
            device=None,
            repeat=False,
            train=True,
            shuffle=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        iterator_unlabeled = data.Iterator(
            dataset=dataset_infer,
            batch_size=len(dataset_infer),
            device=None,
            repeat=False,
            train=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)

        train_preds, train_id = None, []
        unlabeled_preds, unlabeled_id = None, []

        for batch in iterator_train:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            # print(inputs)
            train_id.extend(batch.id.tolist())
            train_preds = self.predict(inputs, target=None, encoding=True)[3]

        for batch in iterator_unlabeled:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            # print(inputs)
            unlabeled_id.extend(batch.id.tolist())
            unlabeled_preds = self.predict(inputs, target=None, encoding=True)[3]

        train_preds = F.normalize(train_preds)
        unlabeled_preds = F.normalize(unlabeled_preds)
        train_num = len(train_id)

        adj = torch.matmul(train_preds, unlabeled_preds.T)
        x, y = torch.where(adj >= threshold)
        semantic_coo = np.eye(8001)
        for i,j in zip(x,y):
            import ipdb
            ipdb.set_trace()
            semantic_coo[train_id[i],unlabeled_id[j]] = 1
        return semantic_coo

    def construct_graph_ac(self, dataset_link, threshold=0.9):

        iterator_train = data.Iterator(
            dataset=dataset_link,
            batch_size=500,
            device=None,
            repeat=False,
            train=True,
            shuffle=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        train_preds = None

        for batch in iterator_train:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            # print(inputs)
            tmp_preds = self.predict(inputs, target=None, encoding=True)[3]
            if train_preds is None:
                train_preds = tmp_preds
            else:
                train_preds = torch.cat([train_preds,tmp_preds],dim=0)

        all_preds = normalize(train_preds.cpu().detach().numpy())

        # tmp_train_id = train_id.copy()
        data_len = len(dataset_link)
        adj = np.matmul(all_preds,all_preds.T)
        x,y = np.where(adj>=threshold)
        semantic_coo = np.eye(data_len,dtype=int)
        # x,y = zip(*[[train_id[i],train_id[j]] for i,j in zip(x.tolist(),y.tolist()) if i in tmp_train_id])
        semantic_coo[x,y] = 1

        return semantic_coo

    # train the model with a batch
    def update(self, inputs, target):
        """ Run a step of forward and backward model update. """
        self.model.train()
        self.optimizer.zero_grad()

        # if self.model_type == 'pointwise':
        #     target = idx_to_onehot(target, self.opt)

        if self.opt['cuda']:
            target = target.cuda()
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])

        logits, _ = self.model(inputs)
        loss = torch.mean(self.criterion(logits, target))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, inputs, target=None, encoding=False):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])
            target = None if target is None else target.cuda()

        self.model.eval()
        logits, encode = self.model(inputs)
        loss = None if target is None else self.criterion(logits, target).mean().data.item()

        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(probs, axis=1).tolist()
        if not encoding:
            return predictions, probs, loss
        else:
            return predictions, probs, loss, encode

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    # save the model
    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),  # model parameters
            'encoder': self.model.encoder.state_dict(),
            'classifier': self.model.classifier.state_dict(),
            'config': self.opt,  # options
            'epoch': epoch,  # current epoch
            'model_type': self.model_type  # current epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    # load the model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.classifier.load_state_dict(checkpoint['classifier'])
        self.opt = checkpoint['config']
        self.model_type = checkpoint['model_type']
        if self.model_type == 'predictor':
            self.criterion = nn.CrossEntropyLoss()
        elif self.model_type == 'pointwise':
            self.criterion = nn.BCEWithLogitsLoss()

class GCN_Trainer(object):
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['gcn_lr'])

    def train(self, features, adj, labels, num_train, num_infer, num_dev):
        t = time.time()
        train_id = list(range(num_train))
        infer_id = list(range(num_train,num_train+num_infer))
        dev_id = list(range(num_train+num_infer,num_train+num_infer+num_dev))
        for epoch in range(self.opt['gcn_epoch']):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(features,adj)
            loss_train = F.nll_loss(output[train_id], labels[train_id])
            acc_train = self.accuracy(output[train_id], labels[train_id])
            loss_train.backward(retain_graph=True)
            self.optimizer.step()

            self.model.eval()
            output = self.model(features, adj)
            loss_val = F.nll_loss(output[dev_id], labels[dev_id])
            acc_val = self.accuracy(output[dev_id], labels[dev_id])
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def retrieve(self, features, adj, iterator_unlabeled, num_train, num_infer, num_dev, k_samples):
        train_id = list(range(num_train))
        infer_id = list(range(num_train, num_train + num_infer))
        dev_id = list(range(num_train + num_infer, num_train + num_infer + num_dev))
        preds = self.model(features, adj)

        meta_idxs = []
        confidence_idxs = []
        examples = iterator_unlabeled.data()
        num_instance = len(examples)

        # ranking
        ranking = list(zip(range(num_instance), preds[infer_id]))
        ranking = sorted(
            ranking, key=lambda x: calc_confidence(x[1], self.opt['beta']), reverse=True)
        # selection
        for eid, pred in ranking:
            if len(meta_idxs) == k_samples:
                break
            rid, _ = arg_max(pred)
            val = calc_confidence(pred, self.opt['alpha'])
            rel = self.opt['rel_itos'][rid]
            meta_idxs.append((eid, rel, examples[eid].relation))
            confidence_idxs.append((eid, val))
        return meta_idxs, confidence_idxs
