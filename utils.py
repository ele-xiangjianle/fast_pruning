import sys
import os
import model.vgg as vgg
import model.resnet_basicblock as resnet_basicblock
import model.resnet_bottleneck as resnet_bottleneck
from collections import OrderedDict
import torch
import torchvision
import os

class Logger():
    def __init__(self):
        self.log_file = None
        self.log_dir = None

    def log(self, str):
        if self.log_file is not None:
            self.log_file.write(str)

    def set_logdir(self, log_dir):
        self.log_dir = log_dir

    def set_filename(self, filename):
        if self.log_dir is not None:
            self.log_file = open(os.path.join(self.log_dir, filename), 'a')

    def close(self):
        if self.log_file is not None:
            self.log_file.close()

def get_model(args):
    if args.type == "vgg":
        if args.arch == 'vgg16':
            return vgg.vgg16()
    if args.type == "basicblock":
        if args.arch == 'resnet20':
            return resnet_basicblock.resnet20()
        if args.arch == 'resnet56':
            return resnet_basicblock.resnet56()
        if args.arch == 'resnet110':
            return resnet_basicblock.resnet110()
        if args.arch == 'resnet164':
            return resnet_basicblock.resnet164()
    if args.type == "bottleneck":
        if args.arch == 'resnet20':
            return resnet_bottleneck.resnet20()
        if args.arch == 'resnet56':
            return resnet_bottleneck.resnet56()
        if args.arch == 'resnet110':
            return resnet_bottleneck.resnet110()
        if args.arch == 'resnet164':
            return resnet_bottleneck.resnet164()
        if args.arch == 'resnet50':
            return torchvision.models.resnet50()
    raise Exception("illegal model arch")

def muti_cuda_to_single(model):
    new_model = OrderedDict()
    for k, v in model.items():
        new_model[k[7:]] = v
    return new_model

# load muti cuda
def load_muti_cuda(model_path, rank, model, optimizer):
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location={
                            'cuda:%d' % 0: 'cuda:%d' % rank})
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return best_acc, start_epoch

def load_model_muti_cuda(model_path, rank):
    checkpoint = torch.load(model_path, map_location={
        'cuda:%d' % 0: 'cuda:%d' % rank})
    return checkpoint['model']

def load_best_acc(model_path):
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    best_acc = checkpoint['acc']
    return best_acc

# load single cuda
def load_model_single_cuda(model_path):
    return torch.load(model_path, map_location='cuda:0')

def load_single_cuda(model_path):
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location='cuda:0')
    acc = checkpoint['acc']
    current_epoch = checkpoint['epoch']
    print('Load checkpoint at epoch {}.'.format(current_epoch))
    print('Accuracy so far {}.'.format(acc))
    return checkpoint['model'], acc

# load cpu
def load_cpu(model_path):
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('Load checkpoint at epoch {}.'.format(start_epoch))
    print('Accuracy so far {}.'.format(best_acc))

# save
def save_state_dict(acc, model, optimizer, current_epoch, model_path):
    checkpoint = {
        'acc': acc,
        'epoch': current_epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path)

def save_model(model_path, model, acc):
    checkpoint = {
        'acc': acc,
        'model': model,
    }
    torch.save(checkpoint, model_path)

if __name__ == "__main__":
    load_cpu(sys.argv[1])
