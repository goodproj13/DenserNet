# coding: utf-8


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os
from glob import glob
import imageio
import argparse

import torch.distributed as dist

from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


arch_train_dir = '../../Arch_train/'
arch_test_dir = '../../Arch_test/'

weight_save_root = "logs"
if not os.path.exists(weight_save_root):
    os.mkdir(weight_save_root)


def synchronize():

    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()



def init_dist_pytorch(args, backend="nccl"):
    args.rank = int(os.environ['LOCAL_RANK'])
    args.ngpus_per_node = torch.cuda.device_count()
    args.gpu = args.rank
    args.world_size = args.ngpus_per_node
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=backend)


class Denser_Net(nn.Module):
  def __init__(self,lower_branch,middle_branch,higher_branch,args):
    super(Denser_Net,self).__init__()

    self.conv_1=nn.Sequential(
        lower_branch,
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(256, args.branch_1_dim, kernel_size=1) ### 384  # 64
    )
    self.conv_m=nn.Sequential(
        middle_branch,
        nn.UpsamplingNearest2d(scale_factor=4),
        nn.Conv2d(512, args.branch_m_dim, kernel_size=1) ### 512  # 64
    )
    self.conv_h=nn.Sequential(
        higher_branch,
        nn.UpsamplingNearest2d(scale_factor=8),
        nn.Conv2d(512, args.branch_h_dim, kernel_size=1)  ### 256  # 64
    )
 
    self.gap = nn.AdaptiveMaxPool2d(1)
    self.fc=nn.Sequential(
          nn.Linear(in_features=args.branch_1_dim+args.branch_m_dim+args.branch_h_dim, out_features=2048, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=2048, out_features=1000, bias=True)   ### 1000
    )
  def forward(self, x):
    h_x = self.conv_1(x)
    m_x = self.conv_m(x)
    l_x = self.conv_h(x)
    
    out = torch.cat((l_x,m_x,h_x), 1)
    out = self.gap(out)
    out = out.view(out.size(0), -1)
    out=self.fc(out)
    return out


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs, device, args):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)
    
  best_test_acc = 0

  for it in range(epochs):
    t0 = datetime.now()
    train_loss = []
    model.train()
    for inputs, targets in tqdm(train_loader, desc="%d/%d (GPU-%d)" % (it+1, epochs, args.gpu)):
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, targets)
        
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

    model.eval()
    test_loss = []
    n_test_correct = 0.
    n_test_total = 0.
    n_train_correct = 0.
    n_train_total = 0.
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      _, predictions = torch.max(outputs, 1)
      loss = criterion(outputs, targets)
      test_loss.append(loss.item())
      n_test_correct += (predictions == targets).sum().item()
      n_test_total+= targets.shape[0]
    
    test_acc = n_test_correct / n_test_total
    test_loss = np.mean(test_loss)

    synchronize()

    if test_acc > best_test_acc:
      if (args.rank==0):
        torch.save(model.module.conv_1.state_dict(), os.path.join(weight_save_root, "DN_vgg16_conv_1_dim-%d.pth" % args.branch_1_dim))
        torch.save(model.module.conv_m.state_dict(), os.path.join(weight_save_root, "DN_vgg16_conv_m_dim-%d.pth" % args.branch_m_dim))
        torch.save(model.module.conv_h.state_dict(), os.path.join(weight_save_root, "DN_vgg16_conv_h_dim-%d.pth" % args.branch_h_dim))
        print("model weights are saved to DN_vgg16_conv_1_dim-%d.pth, DN_vgg16_conv_m_dim-%d.pth, DN_vgg16_conv_h_dim-%d.pth" % (args.branch_1_dim, args.branch_m_dim, args.branch_h_dim) )
      best_test_acc = test_acc

    train_loss = np.mean(train_loss) 

    if it % args.test_epoch != 0:
      continue
    
    
    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      _, predictions = torch.max(outputs, 1)
      n_train_correct += (predictions == targets).sum().item()
      n_train_total+= targets.shape[0]

    synchronize()
    
    train_acc = n_train_correct / n_train_total

    
    train_losses[it] = train_loss
    test_losses[it] = test_loss
    
    dt = datetime.now() - t0
    print('Epoch %d/%d, Train Loss: %f, Train Acc:%f,    Test Loss: %f, Test Acc:%f' % (it+1, epochs, train_loss, train_acc, test_loss, test_acc))
    
  
  return train_losses, test_losses


def main_worker(args):
    global start_epoch, best_recall5
    init_dist_pytorch(args)
    synchronize()

    print("Use GPU: {} for training, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    if (args.rank==0):
        print("==========\nArgs:{}\n==========".format(args))



    train_transform = transforms.Compose([
            transforms.Resize(size=(args.height, args.width)),
            transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
            transforms.Resize(size=(args.height, args.width)),
            transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        arch_train_dir,
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        arch_test_dir,
        transform=train_transform
    )

    if (args.rank==0):
        print("train dataset size:", len(train_dataset.imgs))

    train_data_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=dist.get_rank())
    test_data_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=dist.get_rank())

    batch_size = args.test_batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_data_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_data_sampler
    )


    pre_model = models.vgg16(pretrained=True)
    features=pre_model.classifier[0].in_features

    lower_branch=pre_model.features[:17] ### 16,16-- 2
    middle_branch=pre_model.features[:24] ### 8,8-- 4
    higher_branch=pre_model.features ### 4,4-- 8
    for param in lower_branch.parameters():
        param.requires_grad = False
    for param in middle_branch.parameters():
        param.requires_grad = False
    for param in higher_branch.parameters():
        param.requires_grad = False

    denser_net = Denser_Net(lower_branch,middle_branch,higher_branch,args)

    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    denser_net.cuda(args.gpu)
    denser_net = DistributedDataParallel(denser_net, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, denser_net.parameters()), lr=0.0001,     betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)


    train_losses, test_losses = batch_gd(
        denser_net,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        epochs=100,
        device=device,
        args=args
    )



def main():
    args = parser.parse_args()
    main_worker(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NetVLAD/SARE training")
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')

    parser.add_argument('--branch-1-dim', type=int, default=64)
    parser.add_argument('--branch-m-dim', type=int, default=64)
    parser.add_argument('--branch-h-dim', type=int, default=64)

    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")

    parser.add_argument('--test-epoch', type=int, default=5)
    parser.add_argument('--test-batch-size', type=int, default=16,
                        help="tuple numbers in a batch")

    main()

