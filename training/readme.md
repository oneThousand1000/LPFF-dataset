Please following **[eg3d](https://github.com/NVlabs/eg3d)** and **[stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)** for environment installation respectively.

## Code

Clone our eg3d and stylegan2-ada-pytorch repos.

```
cd training
git clone git@github.com:oneThousand1000/eg3d.git
git clone git@github.com:oneThousand1000/stylegan2-ada-pytorch.git
```

## Training

### EG3D

```
cd eg3d/eg3d
```

**New options:**

**`--camera_sample_mode`**

`camera_sample_mode` is the dataset used to sample the camera for training.

**`--gen_pose_cond_avg`**

When set `--gen_pose_cond_avg=True` and `--gen_pose_cond=False`, the generator will be conditioned on the average camera.



**var1-64.pkl**

```
python train.py --outdir=./training-runs/var1-64 \
    --cfg=ffhq --data=./dataset/eg3d_dataset.zip  \
    --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True  \
    --camera_sample_mode=FFHQ_LPFF
```

**var1-128.pkl**

```
python train.py --outdir=./training-runs/var1-128\
    --cfg=ffhq --data=./dataset/eg3d_dataset.zip  \
    --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128 --kimg=20000 \
    --resume=var1-64.pkl  \
    --camera_sample_mode=FFHQ_LPFF 
```

**var2-64.pkl**

```
python train.py --outdir=./training-runs/var2-64 \
    --cfg=ffhq --data=./dataset/eg3d_dataset.zip  \
    --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True  --gpc_reg_prob=0.8 --kimg=20000  \
    --resume=var1-64.pkl \
    --camera_sample_mode=FFHQ_LPFF_rebalanced 
```

**var2-128.pkl**

```
python train.py --outdir=./training-runs/var2-128 \
    --cfg=ffhq --data=./dataset/eg3d_dataset.zip  \
    --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True  --gpc_reg_prob=0.8 --neural_rendering_resolution_final=128 --kimg=20000\
    --resume=var2-64.pkl \
    --camera_sample_mode=FFHQ_LPFF_rebalanced  
```

**var3-64.pkl**

```
python train.py --outdir=./training-runs/var3-64 \
    --cfg=ffhq --data=./dataset/eg3d_dataset.zip  \
    --gpus=8 --batch=32 --gamma=1 \
    --gen_pose_cond=False --gen_pose_cond_avg=True \ 
    --kimg=20000  \
    --resume=var1-64.pkl \
    --camera_sample_mode=FFHQ_LPFF 
```

**var3-128.pkl**

```
python train.py --outdir=./training-runs/var3-128 \
    --cfg=ffhq --data=./dataset/eg3d_dataset.zip  \
    --gpus=8 --batch=32 --gamma=1 \
    --gen_pose_cond=False --gen_pose_cond_avg=True \
    --neural_rendering_resolution_final=128 --kimg=20000  \
    --resume=var3-64.pkl \
    --rebalance=FFHQ_LPFF  
```

### StyleGAN

```
cd stylegan2-ada-pytorch
```

**FFHQ_LPFF.pkl**

```
python train.py --outdir=./training-runs/FFHQ_LPFF \
	--data=./dataset/stylegan_dataset.zip \
	--gpus=8 --cfg=stylegan2 --mirror=1 --kimg=35000 \
	--camera_sample_mode=FFHQ_LPFF
```

**FFHQ_LPFF_rebalanced.pkl**

```
python train.py --outdir=./training-runs/FFHQ_LPFF_rebalanced \
	--data=./dataset/stylegan_dataset.zip \
	--gpus=8 --cfg=stylegan2 --mirror=1 --kimg=35000\
	--resume=FFHQ_LPFF.pkl\
	--camera_sample_mode=FFHQ_LPFF_rebalanced
```



## EG3D FID Evaluation

```
cd eg3d
cd eg3d
```

**`--camera_sample_mode`**

`camera_sample_mode` is the dataset used to sample rendering camera ( $c_r$ in our main paper). Notice the feature stats for the dataset are computed using the same dataset as `camera_sample_mode` (e.g., when set `--camera_sample_mode=FFHQ`, we will use the FFHQ dataset to compute the dataset feature).

**`--conditional_camera_sample_mode`**

`conditional_camera_sample_mode` is the dataset used to sample the conditional camera ( $c_g$ in our main paper). If you want to fix $c_g$ as the average camera, please use `--conditional_camera_sample_mode=avg`. If you want to set $c_g = c_r$, please use `--conditional_camera_sample_mode=None`. If you want to sample $c_g$ from a certain dataset, please use `--conditional_camera_sample_mode=LPFF` or  `--conditional_camera_sample_mode=FFHQ`.

### Usage

$c_g =c_{avg}, \quad c_r \sim FFHQ$

```
python calc_metrics.py  --metrics=fid50k_full \
	--data=./dataset/eg3d_dataset.zip \  
	--network=network.pkl \  
	--gpus=8  \
	--conditional_camera_sample_mode=avg --camera_sample_mode=FFHQ
```

$c_g =c_{avg}, \quad c_r \sim LPFF$

```
python calc_metrics.py  --metrics=fid50k_full \
	--data=./dataset/eg3d_dataset.zip \  
	--network=network.pkl \  
	--gpus=8  \
	--conditional_camera_sample_mode=avg --camera_sample_mode=LPFF
```

$c_g \sim FFHQ, \quad   c_r \sim FFHQ$

```
python calc_metrics.py --metrics=fid50k_full\
	--data=./dataset/eg3d_dataset.zip \
	--network=network.pkl \ 
	--gpus=8 \
	--conditional_camera_sample_mode=FFHQ --camera_sample_mode=FFHQ 
```

$c_g \sim FFHQ, \quad  c_r \sim LPFF$

```
python calc_metrics.py --metrics=fid50k_full \
	--data=./dataset/eg3d_dataset.zip \
	--network=network.pkl \ 
	--gpus=8 \
	--conditional_camera_sample_mode=FFHQ --camera_sample_mode=LPFF
```

$c_g \sim LPFF, \quad c_r \sim FFHQ$

```
python calc_metrics.py --metrics=fid50k_full  \
	--data=./dataset/eg3d_dataset.zip \
	--network=network.pkl \ 
	--gpus=8 \
	--conditional_camera_sample_mode=LPFF --camera_sample_mode=FFHQ 
```

$c_g \sim LPFF, \quad c_r \sim LPFF$

```
python calc_metrics.py --metrics=fid50k_full \
	--data=./dataset/eg3d_dataset.zip \
	--network=network.pkl \ 
	--gpus=8 \
	--conditional_camera_sample_mode=LPFF --camera_sample_mode=LPFF
```

$c_g \sim FFHQ, \quad  c_r =c_g$

```
python calc_metrics.py --metrics=fid50k_full \
	--data=./dataset/eg3d_dataset.zip \
	--network=network.pkl \ 
	--gpus=8 \
	--camera_sample_mode=FFHQ
```

$c_g \sim LPFF, \quad  c_r =c_g$

```
python calc_metrics.py --metrics=fid50k_full \
	--data=./dataset/eg3d_dataset.zip \
	--network=network.pkl \ 
	--gpus=8 \
	--camera_sample_mode=LPFF
```









