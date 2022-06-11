# Jittor version of AnimeGanV2



## Requirement

Windows 10

python >=3.8

cuda = 10.2

jittor



To install jittor , you can use the following commands:

```
conda install pywin32
python -m pip install jittor
python -m jittor.test.test_core
python -m jittor.test.test_example
python -m jittor.test.test_cudnn_op
```

You can refer to https://cg.cs.tsinghua.edu.cn/jittor/tutorial/ for more details



## Training

Dataset is acquired from [Link](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)(the same as the dataset in pytorch version).
The hyperparameters are tuned according to the official tensorflow implementation of [AnimeGANV2](https://github.com/TachibanaYoshino/AnimeGANv2)
Use `python train.py` to train AneimeGAN.

If your want to tune the hyperparameters, use `python train.py --help` for more details.

## Testing

Use `python test.py` to generate anime style photos.

If your want to tune the hyperparameters, use `python test.py --help` for more details.