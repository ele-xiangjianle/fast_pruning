from email.policy import default
import pruner
import torchvision
import torch
import torch.nn as nn
from thop import profile
import pandas as pd
import argparse
import time
import os
import model.resnet_basicblock as resnet_basicblock
import model.resnet_bottleneck as resnet_bottleneck
from utils import muti_cuda_to_single
from utils import get_model
from utils import load_model_single_cuda
from utils import load_single_cuda
from utils import save_model

LOG_FILE = None
HYPERPARAMS_PRINTED = False


def log_msg(msg):
    print(msg)
    if LOG_FILE is not None:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


def print_pruning_hyperparams(args):
    global HYPERPARAMS_PRINTED
    if HYPERPARAMS_PRINTED:
        return

    HYPERPARAMS_PRINTED = True
    log_msg("=" * 60)
    log_msg("Pruning Configuration:")
    log_msg("=" * 60)
    for k, v in sorted(vars(args).items()):
        log_msg(f"  {k}: {v}")
    log_msg("=" * 60)


def prune(model,args):
    print_pruning_hyperparams(args)
    if args.method == 's-pcc-fdm':
        return prune_pccfdm(model, args)
    
    elif args.method == 'ecp-fdm':
        strategy = pruner.ECPFDMStrategy()
        return prune_strategy(model, args, strategy)

    elif args.method == 'sum-fdm':
        return prune_sumfdm(model, args)

    elif args.method == 'l1':
        strategy = pruner.LNStrategy(p=1, rev=False)
        return prune_strategy(model, args, strategy)

    elif args.method == 'l1-rev':
        strategy = pruner.LNStrategy(p=1, rev=True)
        return prune_strategy(model, args, strategy)

    elif args.method == 'l2':
        strategy = pruner.LNStrategy(p=2, rev=False)
        return prune_strategy(model, args, strategy)

    elif args.method == 'l2-rev':
        strategy = pruner.LNStrategy(p=2, rev=True)
        return prune_strategy(model, args, strategy)

    elif args.method == 'gm':
        strategy = pruner.GMStrategy(rev=False)
        return prune_strategy(model, args, strategy)
    
    elif args.method == 'gm-rev':
        strategy = pruner.GMStrategy(rev=True)
        return prune_strategy(model, args, strategy)

    elif args.method == 'ecp-fdm-rev':
        strategy = pruner.kSCStrategy()
        return prune_strategy(model, args, strategy)
    
    elif args.method == 'random':
        strategy = pruner.RandomStrategy()
        return prune_strategy(model, args, strategy)
    
    elif args.method == 's-pcc-fdm-gurobi':
        return prune_pccfdm_gurobi(model, args)
    
    elif args.method == 's-pcc-fdm-baron':
        return prune_pccfdm_baron(model, args)

    elif args.method == 's-pcc-fdm-cplex':
        return prune_pccfdm_cplex(model, args)

    elif args.method == 's-c-sedp':
        return prune_c_sedp(model, args)



def prune_pccfdm(model, args):
    model.eval()
    # step 1: get all conv
    convs = []
    if args.type == "vgg":
        for conv in model.modules():
            if isinstance(conv, nn.Conv2d):
                convs.append(conv)
    elif args.type == "basicblock":
        for block in model.modules():
            if isinstance(block, resnet_basicblock.BasicBlock):
                convs.append(block.conv1)
    elif args.type == "bottleneck":
        for block in model.modules():
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                convs.append(block.conv1)
                convs.append(block.conv2)

    # step 2: calc the pruned filter count of every conv layer
    strategy = pruner.strategy.PCCFDMStrategy(convs, args.ratio)
    pruned_indices = strategy.run()

    # step 3: make the pruned plans
    dg = pruner.DependencyGraph().build_dependency(
        model, example_inputs=torch.randn(1, 3, 32, 32))
    for i, conv in enumerate(convs):
        plan = dg.get_pruning_plan(
            conv, pruner.prune_conv, pruned_indices[i])
        plan.exec()
    return model

def prune_pccfdm_gurobi(model, args):
    model.eval()
    convs = []
    if args.type == "vgg":
        for conv in model.modules():
            if isinstance(conv, nn.Conv2d):
                convs.append(conv)
    elif args.type == "basicblock":
        for block in model.modules():
            if isinstance(block, resnet_basicblock.BasicBlock):
                convs.append(block.conv1)
    elif args.type == "bottleneck":
        for block in model.modules():
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                convs.append(block.conv1)
                convs.append(block.conv2)

    strategy = pruner.strategy.PCCFDMGurobiStrategy(convs, args.ratio)
    pruned_indices = strategy.run()

    dg = pruner.DependencyGraph().build_dependency(
        model, example_inputs=torch.randn(1, 3, 32, 32))
    for i, conv in enumerate(convs):
        plan = dg.get_pruning_plan(conv, pruner.prune_conv, pruned_indices[i])
        plan.exec()
    return model


def prune_pccfdm_baron(model, args):
    model.eval()
    convs = []
    if args.type == "vgg":
        for conv in model.modules():
            if isinstance(conv, nn.Conv2d):
                convs.append(conv)
    elif args.type == "basicblock":
        for block in model.modules():
            if isinstance(block, resnet_basicblock.BasicBlock):
                convs.append(block.conv1)
    elif args.type == "bottleneck":
        for block in model.modules():
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                convs.append(block.conv1)
                convs.append(block.conv2)

    strategy = pruner.strategy.PCCFDMBaronStrategy(convs, args.ratio)
    pruned_indices = strategy.run()

    dg = pruner.DependencyGraph().build_dependency(
        model, example_inputs=torch.randn(1, 3, 32, 32))
    for i, conv in enumerate(convs):
        plan = dg.get_pruning_plan(conv, pruner.prune_conv, pruned_indices[i])
        plan.exec()
    return model


def prune_pccfdm_cplex(model, args):
    model.eval()
    convs = []
    if args.type == "vgg":
        for conv in model.modules():
            if isinstance(conv, nn.Conv2d):
                convs.append(conv)
    elif args.type == "basicblock":
        for block in model.modules():
            if isinstance(block, resnet_basicblock.BasicBlock):
                convs.append(block.conv1)
    elif args.type == "bottleneck":
        for block in model.modules():
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                convs.append(block.conv1)
                convs.append(block.conv2)

    strategy = pruner.strategy.PCCFDMCPLEXStrategy(convs, args.ratio)
    pruned_indices = strategy.run()

    dg = pruner.DependencyGraph().build_dependency(
        model, example_inputs=torch.randn(1, 3, 32, 32))
    for i, conv in enumerate(convs):
        plan = dg.get_pruning_plan(conv, pruner.prune_conv, pruned_indices[i])
        plan.exec()
    return model


def prune_c_sedp(model, args):
    model.eval()
    convs = []
    if args.type == "vgg":
        for conv in model.modules():
            if isinstance(conv, nn.Conv2d):
                convs.append(conv)
    elif args.type == "basicblock":
        for block in model.modules():
            if isinstance(block, resnet_basicblock.BasicBlock):
                convs.append(block.conv1)
    elif args.type == "bottleneck":
        for block in model.modules():
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                convs.append(block.conv1)
                convs.append(block.conv2)

    strategy = pruner.strategy.C_SEDP(
        convs, args.ratio, beta=args.beta, gamma=args.gamma)
    pruned_indices = strategy.run()

    dg = pruner.DependencyGraph().build_dependency(
        model, example_inputs=torch.randn(1, 3, 32, 32))
    for i, conv in enumerate(convs):
        plan = dg.get_pruning_plan(
            conv, pruner.prune_conv, pruned_indices[i])
        plan.exec()
    return model


def prune_sumfdm(model, args):
    model.eval()
    # step 1: get all conv
    convs = []
    if args.type == "vgg":
        for conv in model.modules():
            if isinstance(conv, nn.Conv2d):
                convs.append(conv)
    elif args.type == "basicblock":
        for block in model.modules():
            if isinstance(block, resnet_basicblock.BasicBlock):
                convs.append(block.conv1)
    elif args.type == "bottleneck":
        for block in model.modules():
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                convs.append(block.conv1)
                convs.append(block.conv2)

    # step 2: calc the pruned filter count of every conv layer
    strategy = pruner.strategy.SUMFDMStrategy(convs, args.ratio)
    pruned_indices = strategy.run()

    # step 3: make the pruned plans
    dg = pruner.DependencyGraph().build_dependency(
        model, example_inputs=torch.randn(1, 3, 32, 32))
    for i, conv in enumerate(convs):
        plan = dg.get_pruning_plan(
            conv, pruner.prune_conv, pruned_indices[i])
        plan.exec()
    return model

def prune_strategy(model, args, strategy):
    model.eval()
    # step 1: get all conv
    convs = []
    if args.type == "vgg":
        for conv in model.modules():
            if isinstance(conv, nn.Conv2d):
                convs.append(conv)
    elif args.type == "basicblock":
        for block in model.modules():
            if isinstance(block, resnet_basicblock.BasicBlock):
                convs.append(block.conv1)
    elif args.type == "bottleneck":
        for block in model.modules():
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                convs.append(block.conv1)
                convs.append(block.conv2)

    # step 2: calc the pruned filter count of every conv layer
    pruned_indices = []
    for conv in convs:
        pruned_indices.append(strategy.apply(conv.weight, args.ratio))

    # step 3: make the pruned plans
    dg = pruner.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 32, 32))
    for i, conv in enumerate(convs):
        plan = dg.get_pruning_plan(conv, pruner.prune_conv, pruned_indices[i])
        plan.exec()

    return model


if __name__ == '__main__':
    model_names = ['vgg16', 'resnet20', 'resnet56',
                   'resnet110', 'resnet164', 'resnet50']
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', default='result/baseline.pth.tar', metavar='PATH',
                        help='input path of baseline')
    parser.add_argument('-o', '--output-path', default='result/baseline-prune.pth.tar', metavar='PATH',
                        help='output path')
    parser.add_argument('-t', '--type', default='vgg', type=str,
                        help='model type (vgg or basicblock or bottleblock)')
    parser.add_argument('-r', '--ratio', default=0.9, type=float,
                        help='ratio')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='temperature beta for soft entropy diversity')
    parser.add_argument('--gamma', default=1.0, type=float,
                        help='temperature gamma for network-level aggregation')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16)')
    parser.add_argument('-d', '--datasets', default='cifar10', type=str,
                        help='datasets')
    parser.add_argument('-m','--method', default='s-pcc-fdm', type=str)
    args = parser.parse_args()

    # ====== 设置日志文件 ======
    log_dir = os.path.dirname(args.output_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.splitext(args.output_path)[0] + '.log'
    log_file = open(log_path, 'w', encoding='utf-8')
    LOG_FILE = log_file

    def log(msg):
        log_msg(msg)

    # ====== 打印参数 ======
    print_pruning_hyperparams(args)

    # ====== 加载模型 ======
    print_pruning_hyperparams(args)
    model = get_model(args)
    if args.datasets == "imagenet":
        state_dict = load_model_single_cuda(args.input_path)
    else:
        state_dict, acc = load_single_cuda(args.input_path)
        state_dict = muti_cuda_to_single(state_dict)

    model.load_state_dict(state_dict)

    example = torch.randn(1, 3, 32, 32)
    ori_flops, ori_params = profile(model, inputs=(example,))

    # ====== 计时剪枝 ======
    log("\nStart pruning...")
    t_start = time.time()
    model = prune(model, args)
    t_end = time.time()
    elapsed = t_end - t_start
    log(f"Pruning finished. Time: {elapsed:.2f}s ({elapsed/60:.2f}min)")

    prune_flops, prune_params = profile(model, inputs=(example,))

    # ====== 输出结果 ======
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame([
        [ori_flops * 2 / 1e6, prune_flops * 2 / 1e6, f"{100*(ori_flops - prune_flops ) / ori_flops }%",
            ori_params / 1e6, prune_params / 1e6,  f"{100*(ori_params - prune_params ) / ori_params }%"]
    ],
        columns=['Flops before prune', 'Flops after prune', 'Flops pruned percent', 'Params before prune', 'Params after prune', 'Params pruned percent'])
    log("\n" + str(model))
    log("\n" + str(df))
    log(f"\nLog saved to: {log_path}")

    # ====== 保存模型 ======
    if args.datasets == "imagenet":
        torch.save(model, args.output_path)
    else:
        save_model(args.output_path, model, acc)

    log_file.close()
