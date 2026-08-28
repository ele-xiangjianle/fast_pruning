import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import os
import torch.distributed as dist
import argparse
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch import optim
import time
import random
from utils import Logger
from utils import get_model
from utils import load_muti_cuda
from utils import load_model_muti_cuda
from utils import load_best_acc
from utils import save_model
from utils import save_state_dict
logger = Logger()


def log(str):
    print(str)
    logger.log(str + '\n')


def train(gpu, args):
    # log
    global logger
    # 模型文件名保持原逻辑（带时间戳）
    args.filename = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
    args.filename = f"{args.filename}-{args.tips}"
    if args.log_dir is not None:
        logger.set_logdir(args.log_dir)
    # 所有 finetune 结果写入同一个日志文件（baseline 训练仍按 tips 命名）
    if args.finetune > 0:
        logger.set_filename("finetune.log")
    else:
        log_name = args.tips if args.tips else args.arch
        logger.set_filename(f"{log_name}.log")

    rank = args.node_rank * args.ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)

    if rank == 0 and args.finetune > 0:
        log("=" * 60)
        log("Finetune Configuration:")
        log("=" * 60)
        log("  time        : {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        log("  type        : {}".format(args.type))
        log("  arch        : {}".format(args.arch))
        log("  prune-ratio : {}".format(args.ratio))
        log("  seed        : {}".format(args.seed))
        log("  beta        : {}".format(args.beta))
        log("  gamma       : {}".format(args.gamma))
        log("=" * 60)

    # data set and sampler
    train_set = torchvision.datasets.CIFAR10(root=args.data,
                                             train=True,
                                             download=False,
                                             transform=transforms.Compose([
                                                 transforms.Pad(4),
                                                 transforms.RandomCrop(32),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010)),
                                             ]))

    test_set = torchvision.datasets.CIFAR10(root=args.data,
                                            train=False,
                                            download=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010))
                                            ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=args.workers,
                              pin_memory=True
                              )

    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             sampler=test_sampler,
                             num_workers=args.workers,
                             pin_memory=True
                             )
    torch.backends.cudnn.benchmark = True
    # close random
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # model criterion
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # best_acc, local_best_acc
    best_acc = 0
    local_best_acc = 0
    # load checkpoint
    if args.resume != '':
        if args.finetune > 0:
            model = load_model_muti_cuda(args.resume, gpu).cuda(gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu], output_device=gpu,  find_unused_parameters=True)
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)

            if os.path.exists(os.path.join(args.output, 'best_checkpoint.pth.tar')):
                best_acc = load_best_acc(os.path.join(
                    args.output, 'best_checkpoint.pth.tar'))
                print('Best accuracy so far {}.'.format(best_acc))
        else:
            model = get_model(args).cuda(gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu], output_device=gpu,  find_unused_parameters=True)
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
            best_acc, args.start_epoch = load_muti_cuda(
                args.resume, gpu,  model, optimizer)
            print('Load checkpoint at epoch {}.'.format(args.start_epoch))
            print('Best accuracy so far {}.'.format(best_acc))
    else:
        model = get_model(args).cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    if args.evaluate:
        correct, total = valid(model, test_loader)
        accuracy = correct / total
        log('Test Acc: {}'.format(accuracy))
        return

    if rank == 0:
        print("ddp backend: {}, world size: {}".format(
            dist.get_backend(), dist.get_world_size()))

    print("rank {} train on cuda {}".format(rank, gpu))
    total_step = len(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train_sampler.set_epoch(epoch)  # shuffle
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.print_freq == 0:
                print('Rank:{} Epoch: [{}/{}], Step: [{}/{}], Loss: {}'.format(
                    rank, epoch + 1, args.epochs, i + 1, total_step, loss.item()))

        # valid
        correct, total = valid(model, test_loader)
        test_result = torch.tensor(
            [correct, total], dtype=torch.float64, device='cuda')
        dist.all_reduce(
            test_result, op=torch.distributed.ReduceOp.SUM, async_op=True)
        torch.distributed.barrier()

        test_result = test_result.tolist()
        all_correct = int(test_result[0])
        all_total = int(test_result[1])
        current_acc = all_correct / all_total

        if rank == 0:
            if args.finetune > 0:
                if current_acc > local_best_acc:
                    local_best_acc = current_acc
                    save_model(os.path.join(args.output, f'{args.filename}.pth.tar'),
                               model, current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    save_model(os.path.join(args.output, f'{args.tips}_best_checkpoint.pth.tar'),
                               model, current_acc)
                print('Current Test Acc: {}'.format(current_acc))
            else:
                save_state_dict(current_acc, model, optimizer, epoch,
                                os.path.join(args.output, 'checkpoint.pth.tar'))
                if current_acc > best_acc:
                    best_acc = current_acc
                    save_state_dict(current_acc, model, optimizer, epoch,
                                    os.path.join(args.output, 'best_checkpoint.pth.tar'))

    if rank == 0:
        print('Local best accuracy so far {}'.format(local_best_acc))
        log('Best accuracy so far {}'.format(best_acc))
        print("completed!")

    if rank == 0 and args.finetune > 0:
        print("[DEBUG] Writing best_acc {:.4f} to log".format(local_best_acc))
        finetune_log_path = os.path.join(args.output, "finetune_results.txt")
        with open(finetune_log_path, "a") as f:
            f.write("{:.6f}\n".format(local_best_acc))
    


def valid(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.cuda(non_blocking=True)
            test_labels = test_labels.cuda(non_blocking=True)
            outputs = model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    return correct, total


def mptrain(args):
    print("Arguments:")
    for k, v in sorted(vars(args).items()):
        print("\t{0}: \t\t\t{1}".format(k, v))
    if args.finetune > 0:
        for _ in range(args.finetune):
            mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))


if __name__ == '__main__':
    model_names = ['vgg16', 'resnet20', 'resnet56', 'resnet110', 'resnet164']
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='./data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=160, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size, this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-tb', '--test-batch-size', default=1024, type=int,
                        metavar='N',
                        help='test-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--node-rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--master-addr', default='localhost', type=str,
                        help='addr used to set up distributed training')
    parser.add_argument('--master-port', default='5678', type=str,
                        help='port used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=2026, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--finetune', default=-1, type=int, metavar='TIMES',
                        help='Fintune the model')
    parser.add_argument('--log-dir', default='./log', type=str,
                        help='the directory used to save logs')
    parser.add_argument('--tips', default=None, type=str,
                        help='logname tips')
    parser.add_argument('--ratio', '--prune-ratio', default=0.0, type=float,
                        help='pruning ratio of the resumed checkpoint')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='pruning temperature beta')
    parser.add_argument('--gamma', default=1.0, type=float,
                        help='pruning temperature gamma')
    parser.add_argument('-t', '--type', default='vgg', type=str,
                        help='model type (vgg or resnet)')
    parser.add_argument('-o', '--output', default='result', type=str,
                        help='output path')
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.nodes
    args.workers = int(
        (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    args.batch_size = int(args.batch_size / args.ngpus_per_node)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    mptrain(args)
