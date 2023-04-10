from data import *
from data.config import custom
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset_root = '/home/broiron/broiron/line_dataset_vol3_pascal/'
cfg = voc

dataset = VOCDetection(root=dataset_root, transform=SSDAugmentation(cfg['min_dim'],
                                                    MEANS), image_sets='train')

base_dir = '/home/broiron/broiron/model_train/ssd_pytorch/'

ssd_net = build_ssd('train', 300, num_classes=2)
net = ssd_net

vgg_weights = torch.load(os.path.join(base_dir, 'weights/vgg16_reducedfc.pth'))
print("loading base network...")
ssd_net.vgg.load_state_dict(vgg_weights)

net = net.to(device)
print(len(dataset))

# setting with default value
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, # 1e-3 -> 1e-5
                          weight_decay=5e-4)
'''
criterion = MultiBoxLoss(num_classes=2, overlap_thresh=0.5, prior_for_matching=True,
                         bkg_label=1, neg_mining=True, neg_pos=3, neg_overlap=0.5,
                         encode_target=False, use_gpu=True)
'''

criterion = MultiBoxLoss(2, 0.5, True, 1, True, 3, 0.5, False, True)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

net.train()

loc_loss = 0
conf_loss = 0
epoch = 0
batch_size = 32

print('loading dataset...')

epoch_size = len(dataset) // batch_size

print('Training SSD on: ', dataset.name)

step_index = 0

data_loader = data.DataLoader(dataset, batch_size, num_workers=4, shuffle=False, collate_fn=detection_collate,
                             pin_memory=True)
print('Data loader length...', len(data_loader))

iter_size = len(data_loader) * epoch_size * 3
print(iter_size)

losses = []

batch_iterator = iter(data_loader)
for iteration in range(0, iter_size):
    loc_loss = 0
    conf_loss = 0
    epoch += 1

    try:
        images, targets = next(batch_iterator)
    except StopIteration:
        batch_iterator = iter(data_loader)
        images, targets = next(batch_iterator)

    # images, targets= next(batch_iterator)
    with torch.no_grad():
        images = Variable(images.to(device))
        targets = [Variable(ann.to(device)) for ann in targets]

    # forward
    t0 = time.time()
    out = net(images)

    # backward
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()

    loc_loss += loss_l.data.item()
    conf_loss += loss_c.data.item()

    if iteration % 1 == 0:
        print('timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')
        losses.append(loss.data.item())

    if iteration != 0 and iteration % 500 == 0:
        print('Saving state, iter:', iteration)
        torch.save(ssd_net.state_dict(), 'weights/ssd300_line2_' +
                       repr(iteration) + '.pth')
torch.save(ssd_net.state_dict(), './weights/linedataset_ver2_1a'+ '.pth')

import pandas as pd

df = pd.DataFrame(losses)
df.to_csv('./losses2.csv')
