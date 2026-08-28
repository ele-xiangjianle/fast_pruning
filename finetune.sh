gamma=1.0

for seed in 2026 2027 2028 2029; do
for beta in 0.1 0.5 1 2 5 10 20 50; do

for ratio in 0.4 0.5 0.6 0.7; do

    python3 trainer.py -a vgg16 -t vgg --resume result/vgg16-prune-${ratio}-b${beta}-g${gamma}.pth.tar \
  --finetune 1 --epochs 40 --tips vgg16-finetune-${ratio}-b${beta}-g${gamma}-s${seed} --output result \
  --ratio ${ratio} --beta ${beta} --gamma ${gamma} --seed ${seed}

done

for ratio in 0.35 0.4 0.45 0.5; do

    python3 trainer.py -a resnet20 -t basicblock --resume result/resnet20-prune-${ratio}-b${beta}-g${gamma}.pth.tar \
  --finetune 1 --epochs 40 --tips resnet20-finetune-${ratio}-b${beta}-g${gamma}-s${seed} --output result \
  --ratio ${ratio} --beta ${beta} --gamma ${gamma} --seed ${seed}

    python3 trainer.py -a resnet56 -t basicblock --resume result/resnet56-prune-${ratio}-b${beta}-g${gamma}.pth.tar \
  --finetune 1 --epochs 40 --tips resnet56-finetune-${ratio}-b${beta}-g${gamma}-s${seed} --output result \
  --ratio ${ratio} --beta ${beta} --gamma ${gamma} --seed ${seed}

done

done
done
