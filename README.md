# Transcending Adversarial Perturbations: Manifold-Aided Adversarial Examples with Legitimate Semantics</p>
[![arXiv](https://img.shields.io/badge/arXiv-2402.03095-red)](https://arxiv.org/pdf/2402.03095.pdf)

> **Transcending Adversarial Perturbations: Manifold-Aided Adversarial Examples with Legitimate Semantics**<br>
> Shuai Li, Xiaoyu Jiang, Xiaoguang Ma <br>
> 
> **Article Status:** Under Preview, **Available source:** Arxiv preprint. <br>
> 
>**Abstract**: <br>
Deep neural networks were significantly vulnerable to adversarial examples manipulated by malicious tiny perturbations. Although most conventional adversarial attacks ensured the visual imperceptibility between adversarial examples and corresponding raw images by minimizing their geometric distance, these constraints on geometric distance led to limited attack transferability, inferior visual quality, and human-imperceptible interpretability. In this paper, we proposed a supervised semantic-transformation generative model to generate adversarial examples with real and legitimate semantics, wherein an unrestricted adversarial manifold containing continuous semantic variations was constructed for the first time to realize a legitimate transition from non-adversarial examples to adversarial ones. Comprehensive experiments on MNIST and industrial defect datasets showed that our adversarial examples not only exhibited better visual quality but also achieved superior attack transferability and more effective explanations for model vulnerabilities, indicating their great potential as generic adversarial examples



## Description
This repo includes the official Pytorch implementation of **Our Method**.

- **Our Method** allows user to realize a legitimate transition from non-adversarial examples to adversarial ones through a supervised semantic-transformation generative model (**SSTGM**).

- ![image](https://drive.google.com/file/d/1UZwHj1e-fGajroOomHGTAxtBkn59X2_R/view?usp=sharing)

- Transitions towards adversarial on manifolds with continuous semantics. By manipulating the two dimensions (`1` and `2`) of the semantic code representation, denoted as `z3`, we presented `5 × 5` image matrices on the left side. 
On the right side, we generated corresponding `5 × 5` heatmaps utilizing the evaluation of `MobileNetV2`. These heatmaps consisted of the prediction results and associated confidences. The images correctly classified by `MobileNetV2`
(non-adversarial) were highlighted with `green squares`, while the misclassified ones (adversarial) were marked with `red squares`.


## Usage

The training and testing experiments are conducted using with a single NVIDIA GeForce RTX 3090 Ti (24 GB).
The memory can be provisioned according to the size of the generated model and the size of the dataset.
1. Prerequisites:
    + Creating a virtual environment in terminal: `conda create -n SSTGM python=3.8`.
    + Installing necessary packages: `pip install -r requirements.txt` (the file of `requirements.txt` contains all the repositories attempted to be used in the experiment, not all of which are required, with the most critical dependency being `Pytorch`).

2. Prepare the dataset and pretrained weights of several well-trained depp-learning models:
    + downloading MNIST dataset from https://pytorch.org/docs/stable/torchvision/datasets.html  
    + downloading NEU-CLS dataset from http://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm
      Assigning your costumed path `--src_image_path` and `path_file_path` to run `Data/process_to_txt.py` in order to create a file of data path.
    + downloading pretrained depp-learning models from [Google Drive](https://drive.google.com/drive/folders/1QUIgzNDf3fqYedien4arsGR8bRubEaol).
      Move them to `Logging/`. 
    + or retraining depp-learning models from scratch through setting `train.py` and running `--CNNs/train.py`, supporting direct training of vgg16, resnet18, mobilenetv2, and resnext50 on two datasets.

3. Prepare the first training stage for generating semantic manipulations of raw data:
   + setting all hyper-parameters according to `Logging/Params/2023-10-08_19-06-09/2023-10-08_19-06-09-hyper-paras.txt` (MNIST) and `Logging/Params/2023-10-24_14-02-52/2023-10-24_14-02-52-hyper-paras.txt` (DEFECT)
   + running the first training stage through executing `--python train.py` (under terminal: `TrainingOnMnist` and `TrainingOnNEU`)
   + observing semantic-oriented results through setting parameter `--save_fig_name_stage`

4. Prepare the second training stage for semantic-transformation generation and launch legitimate attacks:
   + setting all hyper-parameters according to `Logging/Params/2023-10-24_14-02-52/2023-10-24_14-02-52-hyper-paras.txt` (MNIST) and `Logging/Params/2023-10-24_14-02-52/2023-10-24_14-02-52-hyper-paras.txt` (DEFECT)
   + reuning the second training stage through executing `--python train.py` (under terminal: `TrainingOnMnist` and `TrainingOnNEU`)
   + observing semantic-oriented results through setting parameter `--save_fig_name_stage`

4. Prepare the testing for traditional from raw data to adversarial examples:
   + switching to  `TestingThem/TargetM` or `TestingThem/TargetS`
   + assigning your costumed parameters `c_split`
   + starting `AssignTarget(startloop(HOT()))` and executing `--python assign.py` (you can obtain image matrices of semantic-oriented examples and its assessments under specific well-trained deep-learning model)

5. Prepare the testing for comparison of various adversarial examples on visual quality:
   + assigning your costumed parameters for rectangular search, i.e., `step_size` (0.01) and `max_step` (200)
   + starting `AssignTarget(startloop(our_loop()))` and executing `--python assign.py` (you can obtain image matrices of semantic-oriented examples and its assessments under specific well-trained deep-learning model)
   + starting `AssignTarget(startloop(compare_loop(AVERAGE Perturbations)))` and executing `--python assign.py` (you can obtain image matrices of semantic-oriented examples and its assessments under other well-trained deep-learning model)

6. Prepare the testing for comparison of various adversarial examples on attack success rates and attack transferability:
   + starting `AssignTarget(startloop(our_trans_loop()))` and executing `--python assign.py` (you can obtain image matrices of semantic-oriented examples and its assessments under specific well-trained deep-learning model)
   + starting `AssignTarget(startloop(compare_trans_loop(AVERAGE Perturbations)))` and executing `--python assign.py` (you can obtain image matrices of semantic-oriented examples and its assessments under other well-trained deep-learning model)

7. Prepare the testing for comparison of various adversarial examples on breaking known defenses:+ you can obtain two defenses, including adversarial training and network distillation, through `CNNs/train.py`
   + you can obtain two defenses, including adversarial training and network distillation, through `CNNs/train.py`
   + you can test attack success rates after applying defense methods

**The final project should be like this:**

```shell
   SSTGM
   └- CNNs
      └- backbones.py
       └- train.py
       └- trainloops.py
   └- Data
       └- MNIST
       └- NEU-CLS  
   └- Logging
       └- MNIST
       └- NEU-CLS
       └- Params
       └- surface
   └- Results
   └- TestingThem
       └- TargetM
          └- assign.py
          └- ...
       └- TargetS
          └- assign.py
          └- ...
   └- TrainingOnMnist
       └- backbones.py
       └- train.py
       └- trainloops.py
   └- TrainingOnNEU
       └- backbones.py
       └- train.py
       └- trainloops.py
   └- DataM.py
   └- pyproject.toml
   └- DataM.py
   └- requirements.txt
   └- ...
```
