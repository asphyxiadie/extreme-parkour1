# Extreme Parkour with Legged Robots #
<p align="center">
<img src="./images/teaser.jpeg" width="80%"/>
</p>

**Authors**: [Xuxin Cheng*](https://chengxuxin.github.io/), [Kexin Shi*](https://tenhearts.github.io/), [Ananye Agarwal](https://anag.me/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)  
**Website**: https://extreme-parkour.github.io  
**Paper**: https://arxiv.org/abs/2309.14341  
**Tweet Summary**: https://twitter.com/pathak2206/status/1706696237703901439

### Installation ###
```bash
conda create -n parkour python=3.8
conda activate parkour
cd
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone git@github.com:chengxuxin/extreme-parkour.git
cd extreme-parkour
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
# Originally trained with Preview3, but haven't seen bugs using Preview4.
cd isaacgym/python && pip install -e .
cd ~/extreme-parkour/rsl_rl && pip install -e .
cd ~/extreme-parkour/legged_gym && pip install -e .
pip install "numpy<1.24" pydelatin tensorboard tensorboardX tqdm opencv-python ipdb pyfqmr flask
```

### Usage ###
`cd legged_gym/scripts`
To train with the Go2 URDF, select the Go2 task with `--task go2` (for example, `python train.py --task go2 --exptid xxx-xx-go2 --device cuda:0`).

Training metrics are saved as TensorBoard events in `legged_gym/logs/<proj_name>/<exptid>/`. From the repository root, view them with `tensorboard --logdir legged_gym/logs`.

# 先做两轮冒烟训练
python train.py --task go2 --exptid go2-smoke \
  --device cuda:0 --rl_device cuda:0 --debug --max_iterations 2

# Phase 1：训练教师策略
python train.py --task go2 --exptid go2-attn-01 \
  --device cuda:0 --rl_device cuda:0 --num_envs 4096 --max_iterations 15000

# Phase 2：从教师检查点训练视觉学生
python train.py --task go2 --exptid go2-vision-01 \
  --device cuda:0 --rl_device cuda:0 \
  --resume --resumeid go2-attn-01 --delay --use_camera \
  --max_iterations 5000

# 查看训练曲线
tensorboard --logdir ~/extreme-parkour1/legged_gym/logs/parkour_new --port 6006

1. Train base policy:  
```bash
python train.py --exptid xxx-xx-WHATEVER --device cuda:0
```
Train 10-15k iterations (8-10 hours on 3090) (at least 15k recommended).

2. Train distillation policy:
```bash
python train.py --exptid yyy-yy-WHATEVER --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera
```
Train 5-10k iterations (5-10 hours on 3090) (at least 5k recommended). 
>You can run either base or distillation policy at arbitary gpu # as long as you set `--device cuda:#`, no need to set `CUDA_VISIBLE_DEVICES`.

3. Play base policy:
```bash
python play.py --exptid xxx-xx
```
No need to write the full exptid. The parser will auto match runs with first 6 strings (xxx-xx). So better make sure you don't reuse xxx-xx. Delay is added after 8k iters. If you want to play after 8k, add `--delay`

4. Play distillation policy:
```bash
python play.py --exptid yyy-yy --delay --use_camera
```

5. Save models for deployment:
```bash
python save_jit.py --exptid xxx-xx
```
This will save the models in `legged_gym/logs/parkour_new/xxx-xx/traced/`.

conda activate parkour
cd ~/extreme-parkour1/legged_gym/legged_gym/scripts

# Phase 1：base / teacher
python play.py --task go2 --proj_name parkour_new --exptid go2-attn-01 --checkpoint 14500 --delay

# Phase 2：distillation / student（深度相机）
python play.py --task go2 --proj_name parkour_new --exptid go2-vision-01 --checkpoint 9500 --delay --use_camera

lib 11.8问题：
在当前 parkour 环境中执行：
export NVRTC_LIB="$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib"
export LD_LIBRARY_PATH="$NVRTC_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "$LD_LIBRARY_PATH"

### Viewer Usage
Can be used in both IsaacGym and web viewer.
- `ALT + Mouse Left + Drag Mouse`: move view.
- `[ ]`: switch to next/prev robot.
- `Space`: pause/unpause.
- `F`: switch between free camera and following camera.

### Arguments
- --exptid: string, can be `xxx-xx-WHATEVER`, `xxx-xx` is typically numbers only. `WHATEVER` is the description of the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --delay: whether add delay or not.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --use_camera: use camera or scandots.
- --web: used for playing on headless machines. It will forward a port with vscode and you can visualize seemlessly in vscode with your idle gpu or cpu. [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) vscode extension required, otherwise you can view it in any browser.

### Acknowledgement
https://github.com/leggedrobotics/legged_gym  
https://github.com/Toni-SM/skrl

### Citation
If you found any part of this code useful, please consider citing:
```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```
