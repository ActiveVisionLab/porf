# PoRF: Pose Residual Field for Accurate Neural Surface Reconstruction
We present PoRF (pose residual field) for joint optimisation of neural surface reconstruction and camera pose. It uses an MLP to refine the camera poses for all images in the dataset instead of optimising pose parameters for each image independently. The following figure shows that our method can take the COLMAP pose as input and our refined camera pose can be comparable to the GT pose in 3D surface reconstruction. The Chamfer distances (mm) are compared.


![alt tag](https://porf.active.vision/image/dtu_vis.png)



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

- **Example (you need change the address)**

```shell
bash scripts/train_sift_dtu.sh
```

- **Training (without mask)**

```shell
python train.py --mode train --conf confs/dtu_sift_porf.conf --case <case_name>
```

- **Extract surface from the trained model (the first-stage results, only for debugging)** 

```shell
python train.py --mode validate_mesh --conf confs/dtu_sift_porf.conf --case <case_name> --is_continue # use latest checkpoint
```
The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.


- **Export Refined Camera Pose (change folder address)**
```shell
python export_camera_file.py
```


## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{porf_bian2023, 
  title={PoRF: Pose Residual Field for Accurate Neural Surface Reconstruction}, 
  author={Jia-Wang Bian, Wenjing Bian, Victor Adrian Prisacariu, Philip H.S. Torr}, 
  journal= {ICLR}, 
  year={2024} 
}
```

## Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS). Thanks for these great projects.
