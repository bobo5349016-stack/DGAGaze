
DGAGaze is a lightweight gaze estimation framework for gaze prediction in dynamic scenes.

Data Preparation:

The preprocessing pipeline follows the data organization style provided by [GazeHub](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/). Please first prepare the datasets using the corresponding preprocessing tools.

Head Pose Pseudo Labels:

We use [6DRepNet](https://github.com/thohemp/6DRepNet) to generate head-pose pseudo labels for auxiliary supervision.

Training and Evaluation:

To run training and evaluation, use:

python trainer/train.py -s config/train/config_xx.yaml -t config/test/config_xx.yaml

To test your trained model, use:

python tester/test.py -s config/train/config_xx.yaml -t config/test/config_xx.yaml

