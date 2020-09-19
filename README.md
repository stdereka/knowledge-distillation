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
|resnet101_teacher|resnet18_student2|Imagewoof|0.05|7.0|**0.9277**|0.9201| [link](https://github.com/stdereka/knowledge-distillation/tree/d6c45027457de12ae8a5575ff969e70175af708f) |
|resnet101_teacher|resnet18_student2|Imagewoof|0.1|7.0|**0.9308**|0.9201| [link](https://github.com/stdereka/knowledge-distillation/tree/8e4e5f609b095c2a57c59261cdd54501eedb9e15) |



## References

1. https://arxiv.org/pdf/1503.02531.pdf
2. https://arxiv.org/pdf/1812.01819.pdf
3. https://github.com/peterliht/knowledge-distillation-pytorch
