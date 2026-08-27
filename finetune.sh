for ratio in 0.4 0.5 0.6 0.7; do


    python3 trainer.py -a vgg16 -t vgg --resume result/vgg16-prune-${ratio}.pth.tar \
  --finetune 1 --epochs 40 --tips vgg16-finetune-${ratio} --output result
  
done

for ratio in 0.35 0.4 0.45 0.5; do

    python3 trainer.py -a resnet20 -t basicblock --resume result/resnet20-prune-${ratio}.pth.tar \
  --finetune 1 --epochs 40 --tips resnet20-finetune-${ratio} --output result

    python3 trainer.py -a resnet56 -t basicblock --resume result/resnet56-prune-${ratio}.pth.tar \
  --finetune 1 --epochs 40 --tips resnet56-finetune-${ratio} --output result

done