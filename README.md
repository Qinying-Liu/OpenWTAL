# UniWTAL

## Introduction
A unified and simple codebase for weakly-supervised temporal action localization, which currently contains the implementation of ASL(CVPR21), AICL(AAAI23), CASE(ICCV23)

## Install & Requirements
We conduct experiments on the following environment: <br>
* python == 3.6.7 
* pytorch == 1.7.1 
* CUDA == 10.1 

## Data Preparation
1. Download the features of THUMOS dataset from [rec](https://rec.ustc.edu.cn/share/e1472d30-5f38-11ee-a8ae-cff932c459ec). 
2. Place the features inside the `./data` folder.

## Train and Evaluate
1. Train the CASE model by run 
   ```
   python main_case.py --exp_name CASE
   ```
   Train the ASL model by run 
   ```
   python main_asl.py --exp_name ASL
   ```
   Train the AICL model by run 
   ```
   python main_aicl.py --exp_name AICL
   ```
2. The pre-trained model will be saved in the `./outputs` folder. You can evaluate the model by running the command below.
   ```
   python main_case.py --exp_name CASE --inference_only
   ```
   ```
   python main_asl.py --exp_name ASL --inference_only
   ```
   ```
   python main_aicl.py --exp_name AICL --inference_only
   ```

## Todo
1. Code for ActivityNet
2. Code for more methods, e.g., C3BN, BAS-Net

 ## References

* [https://github.com/Pilhyeon/BaSNet-pytorch](https://github.com/Pilhyeon/BaSNet-pytorch)
* [https://github.com/layer6ai-labs/ASL](https://github.com/layer6ai-labs/ASL)

