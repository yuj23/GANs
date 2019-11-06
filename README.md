# Generative Adversarial Network
######  Author:   Li yujue 
## Cycle GAN
[paper link](https://arxiv.org/pdf/1703.10593.pdf)

### Get start
Firstly, you need to run 
``` bash data/download_datasets.sh horse2zebra ```
to get the `horse2zebra`dataset.
the dataset will be downloaded at `data/horse2zebra`with structure

```
horse2zebra
    ├── testA
    ├── testB
    ├── trainA
    └── trainB
```
To run the model

```
python3 train.py
```
The parameters we need are all set in the ```params/params.py```,such like `batch size`,`learing rate`,`dataset path`.


## Reference
the code referenced [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).