# Projection Norm (ProjNorm)

This is the code for the [ICML2022 paper](https://arxiv.org/abs/2202.05834):

### Predicting Out-of-Distribution Error with the Projection Norm 

by Yaodong Yu*, Zitong Yang*, Alexander Wei, Yi Ma, Jacob Steinhardt from UC Berkeley (*equal contribution).

## Prerequisites
* Python
* Pytorch (1.10.0)
* CUDA
* numpy


## How to compute ProjNorm to study model performance under distributional shift?
We use [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) (in-distribution dataset) & [CIFAR10-C](https://arxiv.org/abs/1903.12261) (out-of-distribution datasets) to demonstrate how to compute ProjNorm.

### Step 0: Download OOD data
```bash
mkdir -p ./data/cifar
curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar -C data/cifar/
```

### Step 1: Init base model and reference model
```bash
python init_ref_model.py --arch resnet18 --train_epoch 20 --pseudo_iters 500 --lr 0.001 --batch_size 128 --seed 1
```
#### Arguments:
* ```arch```: network architecture
* ```train_epoch```: number of training epochs for training the base model
* ```pseudo_iters```: number of iterations for training the reference model
* ```lr```: learning rate
* ```batch_size```: mini-batch size
* ```seed```: random seed

#### Output:

The base model (```base_model```) and reference model (```reference_model```) are saved to ```'./checkpoints/{}'.format(arch)```.

### Step 2: Compute ProjNorm for in-distribution data and out-of-distribution data
```bash
python main.py --arch resnet18 --corruption snow --severity 5 --pseudo_iters 500 --lr 0.001 --batch_size 128 --seed 1
```
#### Arguments:
* ```arch```: network architecture (apply the same architecture as in **Step 1**)
* ```corruption```: corruption type
* ```severity```: corruption severity
* ```pseudo_iters```: number of iterations for training the reference model
* ```lr```: learning rate
* ```batch_size```: mini-batch size
* ```seed```: random seed (apply the same random seed as in **Step 1**)

#### Output:

(```in-distribution test error```, ```in-distribution ProjNorm value```)

(```out-of-distribution test error```, ```out-of-distribution ProjNorm value```)

## Reference
For more experimental and technical details, please check our [paper](https://arxiv.org/abs/2202.05834). If you find this useful for your work, please consider citing
```
@article{yu2022predicting,
  title={Predicting Out-of-Distribution Error with the Projection Norm},
  author = {Yaodong Yu and Zitong Yang and Alexander Wei and Yi Ma and Jacob Steinhardt},
  journal={arXiv preprint arXiv:2202.05834},
  year={2022}
}
```
