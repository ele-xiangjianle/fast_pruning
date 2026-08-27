# 1. 训练 VGG16 baseline（160 epochs，DDP）
python3 trainer.py -a vgg16 -t vgg --tips vgg16-baseline --epochs 160
# 训练完后改名
mv result/best_checkpoint.pth.tar result/vgg16-baseline.pth.tar

# 2. 训练 ResNet20 baseline
python3 trainer.py -a resnet20 -t basicblock --tips resnet20-baseline --epochs 160
mv result/best_checkpoint.pth.tar result/resnet20-baseline.pth.tar

# 3. 训练 ResNet56 baseline
python3 trainer.py -a resnet56 -t basicblock --tips resnet56-baseline --epochs 160
mv result/best_checkpoint.pth.tar result/resnet56-baseline.pth.tar