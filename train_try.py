from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
# parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="./config/yolov3-custom-classes.cfg", help="path to model config file")
# parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--train_path", type=str, default="./data/bdd100k_images/bdd100k/images/100k/train", help="path to trainset")
parser.add_argument("--label_path", type=str, default="./data/bdd100k_labels_release/bdd100k/labels/simple_train_labels.json", 
                    help="abs path to train labels")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="./data/class.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data and labels path
train_path = opt.train_path
label_path = opt.label_path
classname_path = opt.class_path

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path, img_size=opt.img_size)
# model.load_weights(opt.weights_path)
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
# Using first 6500 imgs in train folder as trainset
dataloader = torch.utils.data.DataLoader(
    Subset(ListDataset(train_path, label_path, classname_path, img_size=opt.img_size), range(6)), 
    batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=decay, momentum=momentum)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: ((x+1)/burn_in)**2, last_epoch=-1)

for epoch in range(opt.epochs):
    start_time = time.time()        
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        batch_starttime = time.time()
        #update lr in burn_in
        if epoch==0 and batch_i<burn_in:
            scheduler.step()
        
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        
        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()
        
        model.seen += imgs.size(0)
        
        batch_endtime = time.time()
        batchtime = batch_endtime-batch_starttime

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f, Time: %.2fs, lr: %.10f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
                batchtime,
                optimizer.param_groups[0]['lr']
            )
        )
        

    if (epoch+1) % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch+1))

    end_time = time.time()
    epoch_time = end_time-start_time
    print(f'Epoch {epoch+1} done! Time used: {epoch_time:.2f}s')
    print(f'Time on each img: {epoch_time/6:.2f}s')





        
