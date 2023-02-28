# Pre-trained models

## EG3D models

Download link: https://drive.google.com/drive/folders/1eJrXvda9ZwA8NYOLtvr4N-iJ1u9wZ27J?usp=share_link

**<u>Notice:</u>** 

① As explained by the authors of EG3D (please see the [tri-plane issue](https://github.com/NVlabs/eg3d/issues/67)),  [ffhq512-128.pkl](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d/files#) and [ffhqrebalanced512-128.pkl](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d/files#) were achieved using buggy [XY, XZ, ZX] planes. As they suggested in the issue, we trained all of our EG3D-based models **using fixed tri-plane  [XY, XZ, ZY]**. **So be careful if you use set `reload_modules=True` in EG3D generation, please make sure to use the  [XY, XZ, ZY] triplane when you reload modules.**

② var3-64.pkl and var3-128.pkl are finetuned by fixing the camera parameters that are inputted into the generator as $c_g = c_{avg}$. **Please use $c_{avg}$ as the conditional camera to use the two models.**



> **[var1-64.pkl](https://drive.google.com/file/d/1LMclJyg3RcDPBxG9LjaIIXyo938mUcls/view?usp=sharing)** :   The model $E^{Ours}_{var1}$ in our paper, with neural rendering resolution of $64^2$.
>
> **[var1-128.pkl](https://drive.google.com/file/d/1zF_NzBujZwknBkABApcB_lB5usDpvywA/view?usp=sharing)** :  The model $E^{Ours}_{var1}$ in our paper, with neural rendering resolution of  $128^2$.
>
> **[var2-64.pkl](https://drive.google.com/file/d/1fSzmYq1RnoNV4_KWbLFdzg0lPf15jp35/view?usp=sharing)** :  he model $E^{Ours}_{var2}$ in our paper, with neural rendering resolution of $64^2$.
>
> **[var2-128.pkl](https://drive.google.com/file/d/1IEmRhhcgf1uHvLTsVLSZWnJAWNokneAZ/view?usp=sharing)** :  The model $E^{Ours}_{var2}$ in our paper, with neural rendering resolution of $128^2$. 
>
> **[var3-64.pkl](https://drive.google.com/file/d/1umVOcTtV2aCPSFAq0Zvob6DxMZ2WHHGp/view?usp=sharing)** :  he model $E^{Ours}_{var3}$ in our paper, with neural rendering resolution of $64^2$.
>
> **[var3-128.pkl](https://drive.google.com/file/d/1LQUldyLIGSGNe9FgBgnfiT-Qr7FgF7ba/view?usp=sharing)** :  The model $E^{Ours}_{var3}$ in our paper, with neural rendering resolution of $128^2$. 



To provide a fairer comparison, we also retrained EG3D using [XY, XZ, ZY] plane on the FFHQ dataset:

> **[ffhq-fixed-triplane512-64.pkl](https://drive.google.com/file/d/1yJdSI1r5TEf_adFJGNzQki-5wErrxnrl/view?usp=sharing)** : EG3D model trained with FFHQ dataset and has tri-plane fixed (using [XY, XZ, ZY]), with neural rendering resolution of $64^2$.  
>
> **[ffhq-fixed-triplane512-128.pkl](https://drive.google.com/file/d/1MkcgSnyIJHLYb-xjNg2dkLkQZ8k3mK3Z/view?usp=sharing)** : EG3D model trained with FFHQ dataset and has tri-plane fixed (using [XY, XZ, ZY]), with neural rendering resolution of  $128^2$.



### FID

| model                                                        | $c_g =c_{avg}, \\ c_r \sim FFHQ$ | $c_g =c_{avg}, \\ c_r \sim LPFF$ | $c_g \sim FFHQ, \\  c_r \sim FFHQ$ | $c_g \sim FFHQ, \\  c_r \sim LPFF$ | $c_g \sim LPFF, \\ c_r \sim FFHQ$ | $c_g \sim LPFF, \\ c_r \sim LPFF$ | $c_g \sim FFHQ, \\  c_r =c_g$ | $c_g \sim LPFF, \\  c_r =c_g$ |
| ------------------------------------------------------------ | -------------------------------- | -------------------------------- | ---------------------------------- | ---------------------------------- | --------------------------------- | --------------------------------- | ----------------------------- | ----------------------------- |
| $E^{FFHQ}_{var1}$ <br />[ffhq512-128.pkl](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d/files#) | 6.523                            | 23.598                           | 4.273                              | 22.318                             | 23.698                            | 36.641                            | 4.025                         | 23.301                        |
| $E^{FFHQ}_{var1-fixed}$ <br /> [ffhq-fixed-triplane512-128.pkl](https://drive.google.com/file/d/1MkcgSnyIJHLYb-xjNg2dkLkQZ8k3mK3Z/view?usp=sharing) | 7.689                            | 23.962                           | 6.572                              | 22.537                             | 22.567                            | 33.063                            | 6.102                         | 25.115                        |
| $E^{Ours}_{var1}$<br />[var1-128.pkl](https://drive.google.com/file/d/1zF_NzBujZwknBkABApcB_lB5usDpvywA/view?usp=sharing) | 7.997                            | 20.896                           | 6.623                              | 19.738                             | 21.300                            | 22.074                            | 6.093                         | 16.026                        |
| $E^{FFHQ}_{var2}$ <br />[ffhqrebalanced512-128.pkl](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d/files#) | 6.589                            | 20.081                           | 4.456                              | 19.983                             | 19.469                            | 30.181                            | 4.262                         | 23.717                        |
| $E^{Ours}_{var2}$<br />[var2-128.pkl](https://drive.google.com/file/d/1IEmRhhcgf1uHvLTsVLSZWnJAWNokneAZ/view?usp=sharing) | 9.829                            | 16.775                           | 6.672                              | 15.047                             | 13.022                            | 14.836                            | 6.571                         | 12.221                        |
| $E^{Ours}_{var3}$<br />[var3-128.pkl](https://drive.google.com/file/d/1LQUldyLIGSGNe9FgBgnfiT-Qr7FgF7ba/view?usp=sharing) | 6.536                            | 15.852                           | /                                  | /                                  | /                                 | /                                 | /                             | /                             |



## StyleGAN2-ada models

Download link: https://drive.google.com/drive/folders/1N4Tx5AAECueV3YEgDl0YpuRT7QcMWGk3?usp=sharing

> **[FFHQ_LPFF.pkl](https://drive.google.com/file/d/1_iH8swQ9jIn9vXdXnXfeiArFVGAlVFsO/view?usp=sharing)** :  The model $S^{Ours}_{var1}$ in our paper. 
>
> **[FFHQ_LPFF_rebalanced_maxsize89590.pkl](https://drive.google.com/file/d/1d5fjpry5m4e4NKRHQzT-waNMxENcjGXz/view?usp=sharing)** :  The model $S^{Ours}_{var2}$ in our paper. We mistakenly achieved this model using `max_size = 89590`. Notice that the dataset was shuffled before being clipped to `size=89590`, so the pose distribution are not affected much in this pretrained model. We additionally achieved a model using the entire dataset, please see below. 
>
> **[FFHQ_LPFF_rebalanced.pkl](https://drive.google.com/file/d/1KHgCbBh1dd2vg_F-YPLtU4hVmQ0YjXfr/view?usp=sharing)** <u>**(recommended)**</u>: This model was achieved using the same training strategy and dataset as `FFHQ_LPFF_rebalanced_maxsize89590.pkl`, but without a max size limit.

