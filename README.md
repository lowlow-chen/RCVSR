# *Multi-Reference-Based Cross-Scale Feature Fusion for Compressed Video Super Resolution* 

- [*Multi-Reference-Based Cross-Scale Feature Fusion for Compressed Video Super Resolution* ]
  - [0. Background](#0-background)
  - [1. Pre-request](#1-pre-request)
    - [1.1. Environment](#11-environment)
    - [1.2. DCNv2](#12-dcnv2)
    - [1.3. mfqe dataset](#13-mfqe-dataset)
  - [2. Train](#2-train)
  - [3. Test](#3-test)
  - [4. License & Citation](#4-license--citation)


## 0. Background

PyTorch implementation of [*Multi-Reference-Based Cross-Scale Feature Fusion for Compressed Video Super Resolution*](https://ieeexplore.ieee.org/abstract/document/10556779) 

Feel free to contact: <luchen1205@qq.com>.

## 1. Pre-request

### 1.1. Environment

- UBUNTU 20.04/18.04
- CUDA 10.1
- PYTORCH 1.6
- packages: TQDM, LMDB, PYYAML, OPENCV-PYTHON, SCIKIT-IMAGE

Suppose that you have installed CUDA 10.1, then:

```bash
git clone --depth=1 https://github.com/lowlow-chen/RCVSR
conda create -n stdf python=3.7 -y
conda activate rcvsr
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

### 1.2. DCNv2

**Build DCNv2.**

```bash
cd ops/dcn/
bash build.sh
```

**(Optional) Check if DCNv2 works.**

```bash
python simple_check.py
```

> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility. [[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)

### 1.3. MFQEv2 dataset

<details>
<summary><b>Download the raw dataset</b></summary>
<p>
> MFQEv2 dataset includes 108 lossless YUV sequences for training, and 18 test sequences recommended by ITU-T.

</p>
</details>

<details>
<summary><b>Compress sequences</b></summary>
<p>

We now compress both training and test sequences by HM16.5 at LDP mode, QP=27-42.
we will get:

```tex
mfqe/
├── train_raw_yuv/
│── train_qp_yuv/
│   ├── QP32/  
│── train_qp_down/
│   ├── QP27/  
├── test_raw_yuv/
│── test_qp_yuv/
│   ├── QP32/  
│── test_qp_down/
│   ├── QP27/ 
└── README.md
```

</p>
</details>

<details>
<summary><b>Edit YML</b></summary>
<p>

We now edit `option_R3_mfqev2_2G.yml`.

Suppose the folder `mfqe/` is placed at `/home/***/mfqe/`, then you should assign `/home/***/mfqe/` to `dataset -> train -> root` in YAML.

> `R3`: one of the network structures provided in the paper; `mfqev2`: MFQEv2 dataset will be adopted; `2G`: 2 GPUs will be used for the below training. Similarly, you can also edit `option_R3_mfqev2_1G.yml` if needed.

</p>
</details>

<details>
<summary><b>Generate LMDB</b></summary>
<p>

We now generate LMDB to speed up IO during training.

```bash
python create_lmdb_mfqev2.py --opt_path option_R3_mfqev2_2G.yml
```

Now you will get all needed data:

```tex
mfqe/
├── train_raw_yuv/
│── train_qp_yuv/
│   ├── QP32/  
│── train_qp_down/
│   ├── QP27/  
├── test_raw_yuv/
│── test_qp_yuv/
│   ├── QP32/  
│── test_qp_down/
│   ├── QP27/ 
└── README.md
├── LDP_train_gt_qp27.lmdb/
└── LDP_train_lq_qp27.lmdb/
```

Finally, the mfqe dataset root will be sym-linked to the folder `./data/` automatically.

> So that we and programmes can access mfqe dataset at `./data/` directly.

</p>
</details>

## 2. Train
```bash
# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_R3_mfqev2_2G.yml

# 1 GPU
#CUDA_VISIBLE_DEVICES=0 python train.py --opt_path option_R3_mfqev2_1G.yml
```

## 3. Test

Pretrained models can be found here: update later...

<details>
<summary><b>Test MFQEv2 dataset after training</b></summary>
<p>

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --opt_path option_R3_mfqev2_1G.yml
```

</p>
</details>

<details>
<summary><b>Test MFQEv2 dataset without training</b></summary>
<p>

If you did not run `create_lmdb` for training, you should first sym-link MFQEv2 dataset to `./data/`.

```bash
mkdir data/
ln -s /your/path/to/mfqe/ data/MFQEv2
```

</p>
</details>

<details>
<summary><b>Test your own video</b></summary>
<p>

First download the pre-trained model, and then run:

```bash
CUDA_VISIBLE_DEVICES=0 python test_one_video.py
```

See `test_one_video.py` for more details.

</p>
</details>


## 4. License & Citation

You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing the following paper** and **indicating any changes** that you've made.

```tex
@article{RCVSR,
  title={Multi-Reference-Based Cross-Scale Feature Fusion for Compressed Video Super Resolution},
  author={Chen, Lu and Ye, Mao and Ji, Luping and Li, Shuai and Guo, Hongwei},
  journal={IEEE Transactions on Broadcasting},
  year={2024},
  publisher={IEEE}
}
```

