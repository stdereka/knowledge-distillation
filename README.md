# knowledge-distillation

## Requirements

* Python 3.6.9
* CUDA 10.1
* Nvidia Driver 418.67
* Python packages listed in `requirements.txt`

## Experiments

|   Teacher Model   |   Student Model   |  Dataset  | Alpha | T | Accuracy (Distilled) | Accuracy (Only Student) | Code |
|:-----------------:|:-----------------:|:---------:|:-----:|:-:|:--------------------:|:-----------------------:|:----:|
| resnet101_teacher | resnet18_student2 | Imagewoof |  0.05 |7.0|  **0.9247** |    0.9165      |   [link](https://github.com/stdereka/knowledge-distillation/tree/04337ce3037bbfbaed0d0a229cbfbbb235e57b7d)   |
|                   |                   |           |       |   |                      |                         |      |
