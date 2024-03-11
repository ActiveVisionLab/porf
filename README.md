# PoRF: Pose Residual Field for Accurate Neural Surface Reconstruction
We present a novel method for joint optimisation of neural surface reconstruction and camera pose, called PoRF (pose residual field), which uses an MLP to learn the pose residual instead of optimising camera pose parameters directly.


## [Project page](https://porf.active.vision/) |  [Paper](https://arxiv.org/abs/2310.07449) | [Data](https://1drv.ms/u/s!AiV6XqkxJHE2pme7CIkceyLGsng2?e=6qsnlt)
This is the official repo for the implementation of **PoRF: Pose Residual Field for Accurate Neural Surface Reconstruction**.

## Usage

#### Data Convention
The data is organized as follows:

```
<case_name>
|-- cameras.npz        #  GT camera parameters
|-- cameras_colmap.npz #  COLMAP camera parameters 
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- colmap_matches
    |-- 000000.npz        # matches exported from COLMAP
    |-- 000001.npz
    ...
```

Here the `cameras.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Setup

Clone this repository

```shell
git clone https://github.com/ActiveVisionLab/porf.git
cd porf

conda create -n porf python=3.9
conda activate porf
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```


### Running

- **Training (without mask)**

```shell
python train.py --mode train --conf confs/dtu_sift_porf.conf --case <case_name>
```

- **Extract surface from trained model (the first-stage results, only for debug)** 

```shell
python train.py --mode validate_mesh --conf confs/dtu_sift_porf.conf --case <case_name> --is_continue # use latest checkpoint
```

- **Export Refined Camera Pose (change folder address)**
```shell
python export_camera_file.py
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

## Citation

Cite as below if you find this repository is helpful to your project:

```
@inproceedings{bian2024porf, 
  title={PoRF: Pose Residual Field for Accurate Neural Surface Reconstruction}, 
  author={Jia-Wang Bian and Wenjing Bian and Victor Adrian Prisacariu and Philip Torr}, 
  booktitle={ICLR}, 
  year={2024}
}
```

## Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS). Thanks for these great projects.
