# ECE572_Final_Project - Analyzing CAM and GradCAM’s Sensitivity to Different Backdoor Attacks

## Script Description

1. clean_model_training.py
This is used to train a clean resnet18 model on cifar10 dateset.

2. badnets and trojannn.py
This is used to train two backdoor attack models. One is BadNets model, which contains a static trigger, and the other one is TrojanNN model, which contains a dynamic trigger. The target class is "0".

3. visualization.py
This is used to display CAM and GradCam for each model.