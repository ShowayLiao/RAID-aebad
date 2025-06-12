# Inpletation of RAID on AeBAD
The impletation of [DRAE](https://github.com/plutoyuxie/Reconstruction-by-inpainting-for-visual-anomaly-detection) (original [paper](https://www.sciencedirect.com/science/article/pii/S0031320320305094)) on AeBAD<https://github.com/zhangzilongc/MMR> dataset.
We add program in `./datasets/aebad.py` to load the data.
Moreover, the parameters, FLOPS and inference time could be counted in modified `test.py` after setting the `test_paras=True`.

---
## Quick start
Enviroment setting can be seen in [README](origin_README.md). We use `python==3.8, cuda=11.8, pytorch=2.0.0`.
Train on AeBAD_S after changing the dataset path.
```
sh AeBAD_test.sh
sh AeBAD_train.sh
```
---
## Results
inference time per picture:196.31
FLOPS: 77.19G
Parameters: 28.81M
