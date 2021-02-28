from __future__ import absolute_import

import os

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from ..utils.serialization import load_checkpoint, copy_state_dict


__all__ = ['VGG', 'vgg16']


class VGG(nn.Module):
    __factory = {
        16: torchvision.models.vgg16,
    }

    __fix_layers = { # vgg16
        'conv5':24,
        'conv4':17,
        'conv3':10,
        'conv2':5,
        'full':0
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                    train_layers='conv5', log_dir=None, branch_1_dim=64, branch_m_dim=64, branch_h_dim=64):
        super(VGG, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.train_layers = train_layers

        self.branch_1_dim = branch_1_dim
        self.branch_m_dim = branch_m_dim
        self.branch_h_dim = branch_h_dim

        self.feature_dim = self.branch_1_dim + self.branch_m_dim + self.branch_h_dim
        self.log_dir = log_dir
        # Construct base (pretrained) resnet
        if depth not in VGG.__factory:
            raise KeyError("Unsupported depth:", depth)
        vgg = VGG.__factory[depth](pretrained=pretrained)

        lower_branch=vgg.features[:17] ### 16,16-- 2
        middle_branch=vgg.features[:24] ### 8,8-- 4
        higher_branch=vgg.features ### 4,4-- 8

        self.conv_1=nn.Sequential(
            lower_branch,
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, self.branch_1_dim, kernel_size=1)
        )
        self.conv_m=nn.Sequential(
            middle_branch,
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(512, self.branch_m_dim, kernel_size=1) 
        )
        self.conv_h=nn.Sequential(
            higher_branch,
            nn.UpsamplingNearest2d(scale_factor=8),
            nn.Conv2d(512, self.branch_h_dim, kernel_size=1) 
        )

        self.gap = nn.AdaptiveMaxPool2d(1)

        self._init_params()

        if not pretrained:
            self.reset_params()
        else:
            for param in lower_branch.parameters():
                param.requires_grad = False
            for param in middle_branch.parameters():
                param.requires_grad = False
            for param in higher_branch.parameters():
                param.requires_grad = False

            # for param in self.conv_1.parameters():
            #     param.requires_grad = False
            # for param in self.conv_m.parameters():
            #     param.requires_grad = False
            # for param in self.conv_h.parameters():
            #     param.requires_grad = False


    def _init_params(self):
        if (self.log_dir is not None):

            self.conv_1.load_state_dict(torch.load(os.path.join(self.log_dir, 'DN_vgg16_conv_1_dim-%d.pth' % self.branch_1_dim)))
            self.conv_m.load_state_dict(torch.load(os.path.join(self.log_dir, 'DN_vgg16_conv_m_dim-%d.pth' % self.branch_m_dim)))
            self.conv_h.load_state_dict(torch.load(os.path.join(self.log_dir, 'DN_vgg16_conv_h_dim-%d.pth' % self.branch_h_dim)))

            self.pretrained = True

    def forward(self, x):

        h_x = self.conv_1(x)
        m_x = self.conv_m(x)
        l_x = self.conv_h(x)

        x = torch.cat((l_x,m_x,h_x), 1)

        if self.cut_at_pooling:
            return x

        pool_x = self.gap(x)
        pool_x = pool_x.view(pool_x.size(0), -1)

        return pool_x, x

        

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def vgg16(**kwargs):
    return VGG(16, **kwargs)
