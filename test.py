from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import h5py

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed as datadist

from DN import datasets
from DN import models
from DN.evaluators import Evaluator, feature_extraction
from DN.utils.data import IterLoader, get_transformer_train, get_transformer_test
from DN.utils.data.sampler import DistributedSliceSampler
from DN.utils.data.preprocessor import Preprocessor
from DN.utils.logging import Logger
from DN.pca import PCA
from DN.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict, write_json
from DN.utils.dist_utils import init_dist, synchronize


def create_model(args):
    base_model = models.create(args.arch,
        branch_1_dim=args.branch_1_dim, branch_m_dim=args.branch_m_dim, branch_h_dim=args.branch_h_dim)
    if args.vlad:
        pool_layer = models.create('netvlad', dim=base_model.feature_dim)
        model = models.create('embednet', base_model, pool_layer)
    else:
        model = base_model

    model.cuda(args.gpu)
    model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
            )
    return model


def create_data(args):
    root = osp.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, root, scale=args.scale)

    test_transformer_db = get_transformer_test(args.height, args.width)
    test_transformer_q = get_transformer_test(args.height, args.width, tokyo=(args.dataset=='tokyo'))

    pitts = datasets.create('pitts', osp.join(args.data_dir, 'pitts'), scale='30k', verbose=False)
    pitts_train = sorted(list(set(pitts.q_train) | set(pitts.db_train)))
    train_extract_loader = DataLoader(
        Preprocessor(pitts_train, root=pitts.images_dir, transform=test_transformer_db),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(pitts_train),
        shuffle=False, pin_memory=True)

    test_loader_q = DataLoader(
        Preprocessor(dataset.q_test, root=dataset.images_dir, transform=test_transformer_q),
        batch_size=(1 if args.dataset=='tokyo' else args.test_batch_size), num_workers=args.workers,
        sampler=DistributedSliceSampler(dataset.q_test),
        shuffle=False, pin_memory=True)

    test_loader_db = DataLoader(
        Preprocessor(dataset.db_test, root=dataset.images_dir, transform=test_transformer_db),
        batch_size=args.test_batch_size, num_workers=args.workers,
        sampler=DistributedSliceSampler(dataset.db_test),
        shuffle=False, pin_memory=True)

    return dataset, pitts_train, train_extract_loader, test_loader_q, test_loader_db


def test_model(args):
    init_dist(args.launcher, args)
    synchronize()
    cudnn.benchmark = True
    print("Use GPU: {} for testing, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    assert(args.resume)
    if (args.rank==0):
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test_'+args.dataset+'.txt'))
        print("==========\nArgs:{}\n==========".format(args))

    dataset, pitts_train, train_extract_loader, test_loader_q, test_loader_db = create_data(args)

    model = create_model(args)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_recall5 = checkpoint['best_recall5']
        if (args.rank==0):
            print("=> Start epoch {}  best recall5 {:.1%}"
                  .format(start_epoch, best_recall5))


    evaluator = Evaluator(model)
    if (args.reduction):
        pca_parameters_path = osp.join(osp.dirname(args.resume), 'pca_params_'+osp.basename(args.resume).split('.')[0]+'.h5')
        pca = PCA(args.features, (not args.nowhiten), pca_parameters_path)
        if (not osp.isfile(pca_parameters_path)):
            dict_f = feature_extraction(model, train_extract_loader, pitts_train,
                    vlad=args.vlad, gpu=args.gpu, sync_gather=args.sync_gather)
            features = list(dict_f.values())
            if (len(features)>10000):
                features = random.sample(features, 10000)
            features = torch.stack(features)
            if (args.rank==0):
                pca.train(features)
            synchronize()
            del features
    else:
        pca = None

    if (args.rank==0):
        print("Evaluate on the test set:")
    evaluator.evaluate(test_loader_q, sorted(list(set(dataset.q_test) | set(dataset.db_test))),
                        dataset.q_test, dataset.db_test, dataset.test_pos, gallery_loader=test_loader_db,
                        vlad=args.vlad, pca=pca, rerank=args.rerank, gpu=args.gpu, sync_gather=args.sync_gather,
                        nms=(True if args.dataset=='tokyo' else False),
                        rr_topk=args.rr_topk, lambda_value=args.lambda_value)
    synchronize()
    return


def main():
    args = parser.parse_args()

    test_model(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization testing")
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='pitts',
                        choices=datasets.names())
    parser.add_argument('--scale', type=str, default='30k')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")
    parser.add_argument('--num-clusters', type=int, default=64)
    # model
    parser.add_argument('-a', '--arch', type=str, default='vgg16',
                        choices=models.names())
    parser.add_argument('--nowhiten', action='store_true')
    parser.add_argument('--sync-gather', action='store_true')
    parser.add_argument('--features', type=int, default=4096)

    parser.add_argument('--branch-1-dim', type=int, default=64)
    parser.add_argument('--branch-m-dim', type=int, default=64)
    parser.add_argument('--branch-h-dim', type=int, default=64)
    
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--vlad', action='store_true')
    parser.add_argument('--reduction', action='store_true',
                        help="evaluation only")
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--rr-topk', type=int, default=25)
    parser.add_argument('--lambda-value', type=float, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
