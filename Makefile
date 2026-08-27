TIPS=""
TYPE=vgg
# prune
RATIO=0.5
ARCH=vgg16
BASELINE=result/vgg16-baseline.pth.tar
OUTPUT=result
DATASETS=cifar10
METHOD=s-pcc-fdm
OUTPUT_PRUNE = result/vgg16-prune.pth.tar
# train or finetune
RESUME=result/vgg16-prune.pth.tar
EPOCH=40
TIMES=1
LOGDIR=log
.PHONY: train finetune prune finetune-imagenet evaulate

train:
	python3 trainer.py -a $(ARCH) --tips $(TIPS)

evaulate:
	python3 trainer.py -e -a $(ARCH) --resume $(RESUME)

finetune:
	python3 trainer.py -a $(ARCH) -t $(TYPE) --resume $(RESUME) --tips $(TIPS) --finetune $(TIMES) --epoch $(EPOCH) --log-dir $(LOGDIR) --output $(OUTPUT)

finetune-imagenet:
	python3 imagenet-trainer.py -a $(ARCH) --dist-url 'tcp://127.0.0.1:5678' --dist-backend 'nccl' --log-dir $(LOGDIR) --epoch 30 --multiprocessing-distributed --world-size 1 --resume result/resnet50-imagenet-prune.pth.tar --finetune $(TIMES) --rank 0 /oldsys/oldhome/qq/datasets/imagenet

prune:
	python3 prune.py -i $(BASELINE) -o $(OUTPUT_PRUNE) -t $(TYPE) -r $(RATIO) -a $(ARCH) --datasets $(DATASETS) --method $(METHOD)
