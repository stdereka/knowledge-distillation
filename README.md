# knowledge-distillation

This repository contains code for experimenting with knowledge distillation NN compression technique. It was initially proposed in the [article](https://arxiv.org/pdf/1503.02531.pdf). By now pipelines for data processing and model training are implemented for [Imagewoof](https://github.com/fastai/imagenette#imagewoof) dataset, but the framework is compatible with other datasets. Each experiment corresponds to one commit in the repository. This way the code for reproducing the results is easily accesible for every experiment. The most significant results are collected in [Experiment takeaway](#experiment-takeaway) section of this README.

## Contents

* `datasets.py` - contains code to load data in a pytorch compatible format
* `models.py` - models I use in my experiments and distillation loss from the article
* `training.py` - train and evaluation pipelines
* `experiments.ipynb` - contains code, hyperparams and graphics for current experiment
* `report.ipynb` - a detailed explanation of my recent experiments (with visualisation). Can be launched in [Google Colab](https://colab.research.google.com/)

## Requirements

* Python 3.6.9
* CUDA 10.1
* Nvidia Driver 418.67
* Python packages listed in `requirements.txt`

## Install

There are two ways of running the code:

1. In [Google Colab](https://colab.research.google.com/). To start upload `report.ipynb` file to Colab.

2. On local machine. Fulfil [requirements](#requirements) and install Python packages:

        git clone https://github.com/stdereka/knowledge-distillation.git
        cd knowledge-distillation
        pip install -r requirements.txt

## Experiment takeaway

### Temperature search

<table>
<thead>
  <tr>
    <th>Teacher Model</th>
    <th>Student Model</th>
    <th>Dataset</th>
    <th>Alpha</th>
    <th>T</th>
    <th>Accuracy<br>(Distilled)</th>
    <th>Accuracy<br>(No Teacher)</th>
    <th>Code</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="8">resnet101_teacher</td>
    <td rowspan="8">resnet18_student2</td>
    <td rowspan="8">Imagewoof</td>
    <td rowspan="8">0.1</td>
    <td>1.0</td>
    <td>0.9253</td>
    <td rowspan="8">0.9262</td>
    <td rowspan="8"><a href="https://github.com/stdereka/knowledge-distillation/tree/7deaae57bd9f61f70f38f74cf07f5714a6c43932" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td>2.0</td>
    <td>0.9284</td>
  </tr>
  <tr>
    <td>3.0</td>
    <td>0.9298</td>
  </tr>
  <tr>
    <td>4.0</td>
    <td><strong>0.9306</strong></td>
  </tr>
  <tr>
    <td>5.0</td>
    <td>0.9303</td>
  </tr>
  <tr>
    <td>6.0</td>
    <td>0.9295</td>
  </tr>
  <tr>
    <td>7.0</td>
    <td>0.9284</td>
  </tr>
  <tr>
    <td>8.0</td>
    <td>0.9284</td>
  </tr>
</tbody>
</table>

## References

1. https://arxiv.org/pdf/1503.02531.pdf
2. https://arxiv.org/pdf/1812.01819.pdf
3. https://github.com/peterliht/knowledge-distillation-pytorch
4. https://github.com/fastai/imagenette#imagewoof
