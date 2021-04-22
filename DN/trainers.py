from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from .utils.meters import AverageMeter
from tqdm import tqdm

class Trainer(object):

    def __init__(self, model, margin=0.3, gpu=None, temp=0.07):
        super(Trainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.margin = margin
        self.temp = temp

    def _forward(self, inputs, vlad, loss_type):
        B, N, C, H, W = inputs.size()
        inputs = inputs.view(-1, C, H, W)

        outputs_pool, outputs_vlad = self.model(inputs)
        if (not vlad):
            return self._get_loss(outputs_pool, loss_type, B, N)
        else:
            return self._get_loss(outputs_vlad, loss_type, B, N)

    def _get_loss(self, outputs, loss_type, B, N):
        outputs = outputs.view(B, N, -1)
        L = outputs.size(-1)

        output_negatives = outputs[:, 2:]
        output_anchors = outputs[:, 0]
        output_positives = outputs[:, 1]

        if (loss_type=='triplet'):
            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                            margin=self.margin, p=2, reduction='mean')

        elif (loss_type=='sare_joint'):
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist = - torch.cat((dist_pos, dist_neg), 1)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()


        elif (loss_type=='sare_ind'):
            dist_pos = ((output_anchors - output_positives)**2).sum(1)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = ((output_anchors - output_negatives)**2).sum(1)
            dist_neg = dist_neg.view(B, -1)

            dist_neg = dist_neg.unsqueeze(2)
            dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            dist = - torch.cat((dist_pos, dist_neg), 2).view(-1, 2)
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        else:
            assert ("Unknown loss function")

        return loss


    def _parse_data(self, inputs):
        imgs = [input[0] for input in inputs]
        imgs = torch.stack(imgs).permute(1,0,2,3,4)
        # imgs_size: batch_size*triplet_size*C*H*W
        return imgs.cuda(self.gpu)


    def train(self, epoch, sub_id, data_loader, optimizer, train_iters,
                        print_freq=1, vlad=True, loss_type='triplet'):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        data_loader.new_epoch()

        for i in tqdm(range(train_iters), desc="Epoch: [%d-%d] (GPU-%d)" % (epoch, sub_id, self.gpu)):
            inputs = self._parse_data(data_loader.next())
            data_time.update(time.time() - end)

            loss = self._forward(inputs, vlad, loss_type)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0


