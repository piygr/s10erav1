# Session 10 Assignment

1. ResNet architecture for CIFAR10 that has the following architecture:
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    2. Layer1 -
       - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
       - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
       - Add(X, R1)
    3. Layer 2 -
       - Conv 3x3 [256k]
       - MaxPooling2D
       - BN
       - ReLU
    4. Layer 3 -
       - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
       - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
       - Add(X, R2)
    5. MaxPooling with Kernel Size 4
    6. FC Layer 
    7. SoftMax
2. Uses One Cycle Policy such that:
    1. Total Epochs = 24
    2. Max at Epoch = 5
    3. LRMIN = FIND
    4. LRMAX = FIND
    5. NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Use ADAM, and CrossEntropyLoss
6. Target Accuracy: 90%

------
## custom_resnet.py
The file contains the custom resnet model as desired in the assignment. Here is the summary of the network -

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
         Dropout2d-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,856
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
         Dropout2d-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,584
      BatchNorm2d-11          [-1, 128, 16, 16]             256
        Dropout2d-12          [-1, 128, 16, 16]               0
           Conv2d-13          [-1, 128, 16, 16]         147,584
      BatchNorm2d-14          [-1, 128, 16, 16]             256
        Dropout2d-15          [-1, 128, 16, 16]               0
           Conv2d-16          [-1, 256, 16, 16]         295,168
        MaxPool2d-17            [-1, 256, 8, 8]               0
      BatchNorm2d-18            [-1, 256, 8, 8]             512
             ReLU-19            [-1, 256, 8, 8]               0
        Dropout2d-20            [-1, 256, 8, 8]               0
           Conv2d-21            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-22            [-1, 512, 4, 4]               0
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
        Dropout2d-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
        Dropout2d-28            [-1, 512, 4, 4]               0
           Conv2d-29            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-30            [-1, 512, 4, 4]           1,024
        Dropout2d-31            [-1, 512, 4, 4]               0
        MaxPool2d-32            [-1, 512, 1, 1]               0
           Linear-33                   [-1, 10]           5,130
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.38
Params size (MB): 25.08
Estimated Total Size (MB): 32.47
----------------------------------------------------------------
```

## transforms.py
The file contains trasforms which are applied to the input dataset as per the assignment requirement

## dataset.py
CustomCIFAR10Dataset is created on top of CIFAR10 to take care of albumentation + torchvision transforms

## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S10.ipynb
The file is an IPython notebook. The notebook imports helper functions from utils.py.
The LRfinder has been used to find the Max LR. Multiple LR values were tried starting from 0.1 to 1e-8 to find that where the loss is lowest. Based on that the Max LR value is set to 3.20E-04.

## How to setup
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
