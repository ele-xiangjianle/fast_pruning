method=s-c-sedp

for ratio in 0.35 0.4 0.45 0.5 0.6 0.7; do
    # 4. 剪枝 VGG16
    python3 prune.py \
    -i result/vgg16-baseline.pth.tar \
    -o result/vgg16-prune-${ratio}.pth.tar \
    -t vgg -r ${ratio} -a vgg16 \
    --datasets cifar10 --method ${method} \
    --beta 1.0 --gamma 0.5

    # 5. 剪枝 ResNet20
    python3 prune.py \
    -i result/resnet20-baseline.pth.tar \
    -o result/resnet20-prune-${ratio}.pth.tar \
    -t basicblock -r ${ratio} -a resnet20 \
    --datasets cifar10 --method ${method} \
    --beta 1.0 --gamma 0.5

    # 6. 剪枝 ResNet56
    python3 prune.py \
    -i result/resnet56-baseline.pth.tar \
    -o result/resnet56-prune-${ratio}.pth.tar \
    -t basicblock -r ${ratio} -a resnet56 \
    --datasets cifar10 --method ${method} \
    --beta 1.0 --gamma 0.5
done

