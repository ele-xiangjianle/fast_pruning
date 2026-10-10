method=s-c-sedp # 直接取反得到剪枝结果，观察方法本身的有效性
gamma=1.0

for beta in 1; do

for ratio in 0.4 0.5 0.6 0.7; do
    # 4. 剪枝 VGG16
    python3 prune.py \
    -i result/vgg16-baseline.pth.tar \
    -o result/vgg16-prune-${ratio}-b${beta}-g${gamma}-${method}.pth.tar \
    -t vgg -r ${ratio} -a vgg16 \
    --datasets cifar10 --method ${method} \
    --beta ${beta} --gamma ${gamma}
done

for ratio in 0.35 0.4 0.45 0.5; do
    # 5. 剪枝 ResNet20
    python3 prune.py \
    -i result/resnet20-baseline.pth.tar \
    -o result/resnet20-prune-${ratio}-b${beta}-g${gamma}-${method}.pth.tar \
    -t basicblock -r ${ratio} -a resnet20 \
    --datasets cifar10 --method ${method} \
    --beta ${beta} --gamma ${gamma}

    # 6. 剪枝 ResNet56
    python3 prune.py \
    -i result/resnet56-baseline.pth.tar \
    -o result/resnet56-prune-${ratio}-b${beta}-g${gamma}-${method}.pth.tar \
    -t basicblock -r ${ratio} -a resnet56 \
    --datasets cifar10 --method ${method} \
    --beta ${beta} --gamma ${gamma}
done

done
