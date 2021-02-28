import argparse
from torchvision import transforms
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

import pandas as pd
from DN import models
from DN import datasets
from DN.utils.serialization import load_checkpoint, copy_state_dict
from DN.utils.data import IterLoader, get_transformer_train, get_transformer_test
from DN.evaluators import pairwise_distance, extract_features, spatial_nms
from DN.utils.data.preprocessor import Preprocessor
from DN.utils.data.sampler import DistributedSliceSampler
from DN.utils.dist_utils import init_dist, synchronize
from DN.pca import PCA

from collections import OrderedDict

import sys
import os
import numpy as np
import os.path as osp

# img_path = 'sample/image.png'
# model_path = 'pretrained_model/model_best.pth'
# gpu = 0
recall_topk = [1, 5, 10]
result_dir = "result"

def vgg16_netvlad(args, pretrained=False):
    base_model = models.create('vgg16', pretrained=False,
        branch_1_dim=args.branch_1_dim, branch_m_dim=args.branch_m_dim, branch_h_dim=args.branch_h_dim)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    # model = models.create('embednetpca', base_model, pool_layer)
    model = models.create('embednet', base_model, pool_layer)
    model.cuda(args.gpu)
    model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
            )
    if pretrained:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
    return model



def main():
    args = parser.parse_args()

    main_worker(args)

def main_worker(args):
    init_dist(args.launcher, args)
    synchronize()

    print("Use GPU: {} for inference, rank no.{} of world_size {}"
          .format(args.gpu, args.rank, args.world_size))

    if (args.rank==0):
        print("==========\nArgs:{}\n==========".format(args))

    if dist.get_rank() == 0:
        print("inference on '%s'" % args.img_path)
        if os.path.exists(result_dir):
            os.system("rm -rf %s" % result_dir)
        os.mkdir(result_dir)

    # print("Loading model on GPU-%d" % args.gpu)
    model =vgg16_netvlad(args, pretrained=True)
    # read image
    img = Image.open(args.img_path).convert('RGB')
    transformer = transforms.Compose([transforms.Resize((480, 640)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
       std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
    img = transformer(img)
    img = img.cuda(args.gpu)
    # use GPU (optional)
    # model = model.cuda()
    # img = img.cuda()

    # extract descriptor (4096-dim)
    with torch.no_grad():
        outputs = model(img.unsqueeze(0))
        descriptor = outputs[0].cpu()

        if (isinstance(outputs, list) or isinstance(outputs, tuple)):
            x_pool, x_vlad = outputs
            outputs = F.normalize(x_vlad, p=2, dim=-1)
        else:
            outputs = F.normalize(outputs, p=2, dim=-1)

    pca_parameters_path = osp.join(osp.dirname(args.resume), 'pca_params_'+osp.basename(args.resume).split('.')[0]+'.h5')
    pca = PCA(4096, True, pca_parameters_path)

    pca.load(gpu=args.gpu)
    
    if dist.get_rank() == 0:
        outputs = pca.infer(outputs)
        outputs = outputs.data.cpu()

        features = OrderedDict()
        features[args.img_path] = outputs[0]

    root = osp.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, root, scale="30k")
    query = dataset.q_test
    gallery = dataset.db_test

    # distmat, _, _ = pairwise_distance(features, query, gallery)

    test_transformer_db = get_transformer_test(480, 640)
    test_loader_db = DataLoader(
      Preprocessor(dataset.db_test, root=dataset.images_dir, transform=test_transformer_db),
      batch_size=args.test_batch_size, num_workers=args.workers,
      sampler=DistributedSliceSampler(dataset.db_test),
      shuffle=False, pin_memory=True)
    features_db = extract_features(model, test_loader_db, gallery,
      vlad=True, pca=pca, gpu=args.gpu, sync_gather=args.sync_gather)
    synchronize()

    if (dist.get_rank()==0):

        x = torch.cat([features[args.img_path].unsqueeze(0)], 0)
        y = torch.cat([features_db[f].unsqueeze(0) for f, _, _, _ in gallery], 0)

        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
    
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, x, y.t())

        sort_idx = np.argsort(distmat, axis=1)
        del distmat
        db_ids = [db[1] for db in gallery]
        for qIx, pred in enumerate(sort_idx):
            pred = spatial_nms(pred.tolist(), db_ids, max(recall_topk)*12)

            # # comput recall
            # query_id = 
            # correct_at_n = np.zeros(len(recall_topk))
            # gt = dataset.test_pos
            # for i, n in enumerate(recall_topk):
            #     # if in top N then also in top NN, where NN > N
            #     if np.any(np.in1d(pred[:n], gt[query_id])):
            #         correct_at_n[i:] += 1
            #         break
            # recalls = correct_at_n / len(gt)
            # print('Recall Scores:')
            # for i, k in enumerate(recall_topk):
            #     print('  top-{:<4}{:12.1%}'.format(k, recalls[i]))

            # save images
            for i, n in enumerate(recall_topk):
                # if in top N then also in top NN, where NN > N
                result = np.array(gallery)[pred[:n]][:,0]
                # print("top-%d: " % n, result)

                img_save_dir = osp.join(result_dir, "top-%d" % n)
                if not osp.exists(img_save_dir):
                    os.mkdir(img_save_dir)
                for result_img_path in result:
                    os.system("cp %s %s" %(osp.join(dataset.images_dir, result_img_path), img_save_dir))
                print("top-%d image%s saved under '%s'" % (n, "s are" if n > 1 else " is", img_save_dir))

        descriptor = descriptor.numpy()
    
        pd.DataFrame(descriptor).to_csv(os.path.join(result_dir, "descriptor.csv"))
        print("Saved features on CSV")
    
    synchronize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization inference")
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

    parser.add_argument('--img-path', type=str, default='', metavar='PATH')

    main()
