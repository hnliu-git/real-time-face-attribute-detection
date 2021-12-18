## Introduction

A real-time face attribute detection app, built based on Pytorch Lightning and face_recognition.
Recognized attributes can be found in [label.txt](models/labels.txt)
The model is trained on the [Celeba](https://www.kaggle.com/jessicali9530/celeba-dataset) dataset

## How to run

train your own model
- First set your configs in [train.yaml](configs/train.yaml)
- Then run
```shell
python train.py
```

start the app
- Set your configs in [app.yaml](configs/app.yaml)
- Then run
```shell
python app.py
```
