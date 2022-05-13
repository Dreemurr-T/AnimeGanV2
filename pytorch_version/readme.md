## Pytorch version of AnimeGanV2
### Basics
- **Training**<br>
Dataset is acquired from [Link](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)<br>
The hyperparameters are tuned according to the official tensorflow implementation of [AnimeGANV2](https://github.com/TachibanaYoshino/AnimeGANv2)<br>
Use `python train.py` to train AneimeGAN from scratch, to resume checkpoint, use:<br>
```
python train.py --start_epoch <int> --if_resume True
```
If your want to tune the hyperparameters, use `python train.py --help` for more details<br>
- **Testing**<br>
Use `python test.py` to generate anime style photos, check `python test.py --help` for more detals. Hayao style checkpoint will be added later<br>
