# knowledge-distillation

This repository contains code for experimenting with knowledge distillation NN compression technique. It was initially proposed in the [article](https://arxiv.org/pdf/1503.02531.pdf). By now pipelines for data processing and model training are implemented for [Imagewoof](https://github.com/fastai/imagenette#imagewoof) dataset, but the framework is compatible with other datasets. Each experiment corresponds to one commit in the repository. This way the code for reproducing the results is easily acceseble for every experiment. The most significant results are collected in [Experiment takeaway](#experiment-takeaway) section of this README.

## Contents

* `datasets.py` - contains code to load data in a pytorch compatible format
* `models.py` - models I use in my experiments and distillation loss from the article
* `training.py` - train and evaluation pipelines
* `experiments.ipynb` - contains code, hyperparams and graphics for current experiment. Can be launched in [Google Colab](https://colab.research.google.com/)

## Requirements

* Python 3.6.9
* CUDA 10.1
* Nvidia Driver 418.67
* Python packages listed in `requirements.txt`

## Experiment takeaway

|   Teacher Model   |   Student Model   |  Dataset  | Alpha | T | Accuracy (Distilled) | Accuracy (Only Student) | Code |
|:-----------------:|:-----------------:|:---------:|:-----:|:-:|:--------------------:|:-----------------------:|:----:|
| resnet101_teacher | resnet18_student2 | Imagewoof |  0.05 |7.0|  **0.9247** |    0.9165      |   [link](https://github.com/stdereka/knowledge-distillation/tree/04337ce3037bbfbaed0d0a229cbfbbb235e57b7d)   |
|resnet101_teacher|resnet18_student2|Imagewoof|0.05|7.0|**0.9277**|0.9201| [link](https://github.com/stdereka/knowledge-distillation/tree/d6c45027457de12ae8a5575ff969e70175af708f) |
|resnet101_teacher|resnet18_student2|Imagewoof|0.1|7.0|**0.9308**|0.9201| [link](https://github.com/stdereka/knowledge-distillation/tree/8e4e5f609b095c2a57c59261cdd54501eedb9e15) |

## References

1. https://arxiv.org/pdf/1503.02531.pdf
2. https://arxiv.org/pdf/1812.01819.pdf
3. https://github.com/peterliht/knowledge-distillation-pytorch
4. https://github.com/fastai/imagenette#imagewoof
