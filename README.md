
# Facial Emotion Recognition with VGG models
This project trains VGG-11, VGG-16, and a VGG-16 (transfer learning on a pre-trained model) on  various datasets like [FER2013](https://paperswithcode.com/dataset/fer2013), [FER+](https://paperswithcode.com/dataset/fer) and a [custom dataset from Kaggle based on Affectnet](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data), and compare the various model performances.

The models are trained using Pytorch.

Live camera feed and video clips inference using MediaPipe.

## Model Used

### VGG-11

![VGG-11 Architechture](Assets/model_archs/vgg-11-arch.png)

### VGG-16

![VGG-16 Architechture](Assets/model_archs/vgg-16-arch.png)

## Data Visualisation

### FER 2013

![FER2013 class distribution](Assets/datasets/FER2013-train.png)

### FER+

![FER+ class distribution](Assets/datasets/FERplus-train.png)

### Custom Face Data

![CFD class distribution](Assets/datasets/CFD-distribution.png)


## Models Training and Evaluation

### Models Accuracy and Loss
![Models Accuracy](Assets/training/accuracy-graphs.png)
![Models Loss](Assets/training/loss-graphs.png)

### Models Confusion Matrix
![Models Confusion Matrix](Assets/training/confusion-matrix.png)

### Models F1 Score and Accuracy Score
![Models Accuracy Scores](Assets/training/accuracy-plot.png)
![Models F1 Scores](Assets/training/f1-plot.png)

## Inference of the Model

### Video Clips



https://github.com/Saksham1970/emotion-recognition-vgg/assets/45041294/06275671-2592-451f-8536-29754019c683



https://github.com/Saksham1970/emotion-recognition-vgg/assets/45041294/2c87a046-a8a6-4661-b089-abd0670ced2e


### Live Camera Feed

![Camera Example 1](Assets/camera/live1.png)
![Camera Example 2](Assets/camera/live2.png)
![Camera Example 3](Assets/camera/live3.png)
![Camera Example 4](Assets/camera/live4.png)




