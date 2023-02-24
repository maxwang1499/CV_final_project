# CV_final_project
The aim of our computer vision final project is to construct an image classification model that can identify whether a given satellite image captures one of six types of methane-emitting facilities. You can read our paper here: https://tinyurl.com/2bt3dkw4.

Based on repo from Stanford ML Group: https://github.com/stanfordmlgroup/aicc-aut21-multi-task

To train the model, run:

```python main.py train```

To evaluate the model on the test set, run:

```python main.py test --ckpt_path <checkpoint path>```
