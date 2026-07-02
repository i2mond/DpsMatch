# (Under Review)Decoder Pre-training with Synthetic Images for Semi-supervised Semantic Segmentation

## 1. Setup

Download the base repository:

- **UniMatch V2**: https://github.com/LiheYoung/UniMatch-V2

Place the following files and folders inside the `UniMatch-V2` directory:

```
UniMatch-V2/
├── MaskFactory/
├── SegRCDB/
├── configs/
│   ├── MaskFactory.yaml
│   └── SegRCDB.yaml
├── datasets/
│   ├── segrcdb_semi.py
│   └── maskfactory_semi.py
├── dps_segrcdb.py
├── dps_maskfactory.py
├── fine_unimatch_v2.py
├── dps_train.sh
└── fine_train.sh
```

## 2. Dataset Preparation

Download the datasets from their official repositories:

- **MaskFactory**: https://github.com/ydchen0806/MaskFactory
- **SegRCDB**: https://github.com/dahlian00/SegRCDB

Prepare the following folder structures under `UniMatch-V2/`.

### MaskFactory

```
MaskFactory/
├── img/
├── mask/
└── txt/
```

### SegRCDB

```
SegRCDB/
├── image/
├── mask/
└── txt/
```

> Note: This was verified by actually running the generation code. The number of classes is 255 (background: 0, foreground: 1–245).

For `MaskFactory/img`, `MaskFactory/mask`, `SegRCDB/image`, and `SegRCDB/mask`, files should be named sequentially, e.g.:

```
000001.png
000002.png
000003.png
...
```

## 3. Decoder Pretraining

### Pretraining with MaskFactory

Run `dps_maskfactory.py` using `configs/MaskFactory.yaml` as the configuration file.

### Pretraining with SegRCDB

Run `dps_segrcdb.py` using `configs/SeRCDB.yaml` as the configuration file.

### Launching Pretraining (Distributed)

```bash
sh dps_train.sh 2 <port>
```

- `2`: number of GPUs
- `<port>`: port number for distributed training

Our trained weights [here](https://drive.google.com/drive/folders/1sdB-5179PzT9nBoQZ4xXQ2-rVlPHGesm?usp=drive_link).


## 4. Fine-tuning

After decoder pretraining, fine-tune the model using `fine_unimatch_v2.py`. This step loads the decoder weights obtained from the pretraining stage above.

### Launching Fine-tuning (Distributed)

```bash
sh fine_train.sh 2 <port>
```

- `2`: number of GPUs
- `<port>`: port number for distributed training

## Notes

- Adjust the number of GPUs and port number in the shell scripts according to your environment.

## Acknowledge

This implementation is based on UniMatch V2 [1], MaskFactory [2], and SegRCDB [3]. Thanks for the awesome work.

[1] L. Yang, Z. Zhao, and H. Zhao, "Unimatch v2: Pushing the limit of semi-supervised semantic segmentation," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 47, no. 4, pp. 3031–3048, 2025.

[2] H. Qian, Y. Chen, S. Lou, F. S. Khan, X. Jin, and D.-P. Fan, "MaskFactory: Towards high-quality synthetic data generation for dichotomous image segmentation," in *Advances in Neural Information Processing Systems*, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 66455–66478.

[3] R. Shinoda, R. Hayamizu, K. Nakashima, N. Inoue, R. Yokota, and H. Kataoka, "SegRCDB: Semantic segmentation via formula-driven supervised learning," in *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 19997–20006.
