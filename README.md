# GARField

GARField (Garment Attached Radiance Field) adresses the data problem in deformable object manipulation through learnable, viewpoint-free data generation.

[Project website](https://ddonatien.github.io/garfield-website/)

## Installation

``` sh
git clone git@github.com:ddonatien/GARField.git
cd GARField
pip install -r requirements.txt
```

## Training

``` sh
python run.py --mode train_render --conf confs/sock.conf --case sock_cap_0 --gpu 0
```

## Image generation

``` sh
python run.py --mode render_novel --conf confs/sock.conf --case sock_cap_0 --gpu 0  --model <MODEL_FILE>.pth
```

## Citation
```
@article{delehelle2024garfield,
  title={GARField: Addressing the visual Sim-to-Real gap in garment manipulation with mesh-attached radiance fields},
  author={Delehelle, Donatien and Caldwell, Darwin G and Chen, Fei},
  journal={arXiv preprint arXiv:2410.05038},
  year={2024}
}
```
