# [ECCV 2022] Latent Partition Implicit with Surface Codes for 3D Representation

This repository contains the code to reproduce the results from the paper "Latent Partition Implicit with Surface Codes for 3D Representation".

[**Homepage**](https://chenchao15.github.io/LPI_page/) **|**[**Paper**](https://arxiv.org) **|** [**Supp**](https://cloud.tsinghua.edu.cn/f/5d690a9ed5054a8e9416/?dl=1)

If you find our code or paper useful, please consider citing:

    @inproceedings{LPI,
        title = {Latent Partition Implicit with Surface Codes for 3D Representation},
        author = {Chao, Chen and Yu-shen, Liu and Zhizhong, Han},
        booktitle = {European Conference on Computer Vision (ECCV)},
        year = {2022}
    }

### demo

<img src="img/LPI.gif" alt="Webp.net-gifmaker" style="zoom:100%;" />

### Installation

We support ```python3```, you can first create an virture environment called ```LPI_venv ```:

```
python -m venv LPI_venv
source venv/bin/activate
```

Then, to install the dependencies, run:

```
pip install -r requirements.txt
```

Next, for evaluation of the models, complie the extension modules, which are provided by [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks). run:

```
python setup.py build_ext --inplace
```

To compile the dmc extension, you have to have a cuda enabled device set up. If you experience any errors, you can simply comment out the `dmc_*` dependencies in `setup.py`. You should then also comment out the `dmc` imports in `im2mesh/config.py`.

Finally, for calculating chamfer distance faster during training, we use the Customized TF Operator ```nn_distance```,  run:

```
cd nn_distance
./tf_nndistance_compile.sh
```

If you encounter any errors, please refer to the relevant instructions of [PU-net](https://github.com/yulequan/PU-Net) and modify the compile scripts slightly.

### Dataset

To be finished...

### Training

Training and evaluating single 3d object:

```
./run.sh
```

Training and evaluating all 3d objects of a class:

```
./multi_run.sh
```

### Evaluation

Evaluating single 3d object:

```
./test.sh
```

Evaluating all 3d objects of a class:

```
./multi_test.sh
```

To be finished...

#### Deployment instructions are to be finished soon.

