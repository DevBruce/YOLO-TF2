# YOLO with Tensorflow 2

![tf-v2.4.1](https://img.shields.io/badge/TensorFlow-v2.4.1-orange)

For ease of implementation, i have not implemented exactly the same as paper.  
The things presented below are implemented differently from the paper.

- Backbone network. (I used **Xception** instead of network mentioned in the paper.)

- Learning Rate Schedule  
(I used `tf.keras.optimizers.schedules.ExponentialDecay`)

- Hyper Parameters

- Data Augmentations

- And so on . . .

<br><br>

## Preview

Will be updated soon . . .

<br><br>

## Build Environment with Docker

### Build Docker Image

```bash
$ docker build -t ${NAME}:${TAG} .
```

### Create a Container

```bash
$ docker run -d -it --gpus all --shm-size=${PROPER_VALUE} ${NAME}:${TAG} /bin/bash
```

<br><br>

## Training PascalVOC Dataset

- **PascalVOC Dataset** with [TFDS](https://www.tensorflow.org/datasets/overview) (Training Script: [./voc_scripts/train_voc.py](./voc_scripts/train_voc.py))

- Training Set: PascalVOC2007 trainval + PascalVOC2012 trainval
- Validation Set: PascalVOC2007 test

```bash
$ python train_voc.py
```

**Options**  

Default option values are [./configs/configs.py](./configs/configs.py).  
If the options are given, the default config values are overridden.  

- `--epochs`: Number of training epochs
- `--init_lr`: Initial learning rate
- `--lr_decay_rate`: Learning rate decay rate
- `--lr_decay_steps`: Learning rate decay steps
- `--batch_size`: Training batch size
- `--val_step`: Validation interval during training
- `--tb_img_max_outputs `: Number of visualized prediction images in tensorboard
- `--val_sample_num`: Validation sampling. 0 means use all validation set.

<br><br>

## Citation

**You Only Look Once: Unified, Real-Time Object Detection** \<[arxiv link](https://arxiv.org/abs/1506.02640)\>

```
@misc{redmon2016look,
      title={You Only Look Once: Unified, Real-Time Object Detection}, 
      author={Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi},
      year={2016},
      eprint={1506.02640},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
