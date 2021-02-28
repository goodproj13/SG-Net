
# coding: utf-8

# In[1]:


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

# In[2]:

arch_train_dir = '../../Arch_train/'
arch_test_dir = '../../Arch_test/'
# print("Dataset dir: %s %s" % (arch_train_dir, arch_test_dir))

weight_save_root = "logs"
if not os.path.exists(weight_save_root):
    os.mkdir(weight_save_root)


# In[6]:


# conv_1_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 64  ### 384
# conv_m_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 64  ### 512
# conv_h_dim = int(sys.argv[3]) if len(sys.argv) > 2 else 64  ### 256

def init_dist_pytorch(args, backend="nccl"):
    args.rank = int(os.environ['LOCAL_RANK'])
    args.ngpus_per_node = torch.cuda.device_count()
    args.gpu = args.rank
    args.world_size = args.ngpus_per_node
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=backend)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


class Denser_Net(nn.Module):
  def __init__(self,lower_branch,middle_branch,higher_branch,args):
    super(Denser_Net,self).__init__()
    ### add batch norm
    # self.conv_1=nn.Sequential(
    #     lower_branch,
    #     nn.UpsamplingNearest2d(scale_factor=2),
    #     nn.Conv2d(256, args.branch_1_dim, kernel_size=1), ### 384  # 64
    #     nn.BatchNorm2d(args.branch_1_dim)
    # )
    # self.conv_m=nn.Sequential(
    #     middle_branch,
    #     nn.UpsamplingNearest2d(scale_factor=4),
    #     nn.Conv2d(512, args.branch_m_dim, kernel_size=1), ### 512  # 64
    #     nn.BatchNorm2d(args.branch_m_dim)
    # )
    # self.conv_h=nn.Sequential(
    #     higher_branch,
    #     nn.UpsamplingNearest2d(scale_factor=8),
    #     nn.Conv2d(512, args.branch_h_dim, kernel_size=1),  ### 256  # 64
    #     nn.BatchNorm2d(args.branch_h_dim)
    # )
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
    # self.fc=nn.Sequential(
    #       nn.Linear(in_features=1024*(args.branch_1_dim+args.branch_m_dim+args.branch_h_dim), out_features=4096, bias=True),  ### 196608
    #       nn.ReLU(inplace=True),
    #       nn.Dropout(p=0.5, inplace=False),
    #       nn.Linear(in_features=4096, out_features=4096, bias=True),
    #       nn.ReLU(inplace=True),
    #       nn.Dropout(p=0.5, inplace=False),
    #       nn.Linear(in_features=4096, out_features=1000, bias=True)   ### 1000
    # )
    self.gap = nn.AdaptiveMaxPool2d(1)
    self.fc=nn.Sequential(
          nn.Linear(in_features=args.branch_1_dim+args.branch_m_dim+args.branch_h_dim, out_features=2048, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=2048, out_features=1000, bias=True)   ### 1000
    )
  def forward(self, x):
    # print(x.shape)
    h_x = self.conv_1(x)
    m_x = self.conv_m(x)
    l_x = self.conv_h(x)
    
    # ## concat features
    out = torch.cat((l_x,m_x,h_x), 1)
    # out=F.relu(out)
    # out = out.view(out.size(0), -1)# flatten
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
      # move data to GPU
      inputs, targets = inputs.to(device), targets.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, targets)
        
      # Backward and optimize
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

    # torch.save(model.state_dict(), os.path.join(weight_save_path, "model_%d.pth" % (it+1)))
    if test_acc > best_test_acc:
      if (args.rank==0):
        torch.save(model.module.conv_1.state_dict(), os.path.join(weight_save_root, "DN_vgg16_conv_1_dim-%d.pth" % args.branch_1_dim))
        torch.save(model.module.conv_m.state_dict(), os.path.join(weight_save_root, "DN_vgg16_conv_m_dim-%d.pth" % args.branch_m_dim))
        torch.save(model.module.conv_h.state_dict(), os.path.join(weight_save_root, "DN_vgg16_conv_h_dim-%d.pth" % args.branch_h_dim))
        print("model weights are saved to DN_vgg16_conv_1_dim-%d.pth, DN_vgg16_conv_m_dim-%d.pth, DN_vgg16_conv_h_dim-%d.pth" % (args.branch_1_dim, args.branch_m_dim, args.branch_h_dim) )
      best_test_acc = test_acc

    # Get train loss and test loss
    train_loss = np.mean(train_loss) # a little misleading

    if it % args.test_epoch != 0:
      continue
    
    
    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      _, predictions = torch.max(outputs, 1)
      n_train_correct += (predictions == targets).sum().item()
      n_train_total+= targets.shape[0]
    # Save losses
    synchronize()
    
    train_acc = n_train_correct / n_train_total

    
    train_losses[it] = train_loss
    test_losses[it] = test_loss
    
    dt = datetime.now() - t0
    # print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc:{train_acc:.4f},\
    # Test Loss: {test_loss:.4f}, Test Acc:{test_acc:.4f}')
    print('Epoch %d/%d, Train Loss: %f, Train Acc:%f,    Test Loss: %f, Test Acc:%f' % (it+1, epochs, train_loss, train_acc, test_loss, test_acc))
    
  
  return train_losses, test_losses


def main_worker(args):
    global start_epoch, best_recall5
    init_dist_pytorch(args)
    synchronize()

    print("Use GPU: {} for training, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    if (args.rank==0):
        # sys.stdout = Logger(osp.join(args.logs_dir, 'log_feature_branch.txt'))
        print("==========\nArgs:{}\n==========".format(args))


    # In[3]:

    # Note: normalize mean and std are standardized for ImageNet
    # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
    train_transform = transforms.Compose([
            transforms.Resize(size=(args.height, args.width)),
            transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
            transforms.Resize(size=(args.height, args.width)),
            transforms.ToTensor()
    ])

    # print("Loading dataset")
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
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     sampler=train_data_sampler,
    #     shuffle=True
    # )
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


    # In[4]:


    # for inputs,tars in train_loader:
    #     print(inputs[0].shape)
    #     plt.imshow(inputs[0].permute(1,2,0))
    #     print(tars[0])
    #     break
    # K=0
    # # print(os.getcwd())
    # for i in os.listdir(arch_test_dir):
    #     K+=1
    # print(K)


    # In[5]:


    # Define the model
    pre_model = models.vgg16(pretrained=True)
    features=pre_model.classifier[0].in_features

    # pre_model
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
    # print(device)
    # denser_net.to(device)
    denser_net.cuda(args.gpu)
    denser_net = DistributedDataParallel(denser_net, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    # In[9]:


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


    # c=0
    # for i,t in train_loader:
    #     plt.imshow(i[3].permute(1,2,0))
    #     outs=cnn_model(i[3].unsqueeze(0).to(device))
    #     _,pred=torch.max(outs,1)
    #     print(pred == t[3])
    #     plt.title(f'Pred:{pred.cpu().numpy()}---Label:{t[3]}')
    #     break
    


    # # In[ ]:


    # plt.plot(train_losses, label='train loss')
    # plt.plot(test_losses, label='test loss')
    # plt.legend()
    # plt.show()


    # # In[ ]:


    # torch.save(cnn_model.state_dict(),"trained_arc.pt")


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

