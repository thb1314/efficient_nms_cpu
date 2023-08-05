# NMS Library

NMS(Non Maximum Suppression) is a computer vision method that selects a single entity out of many overlapping entities (for example bounding boxes in object detection).

This project provides and compares a variety of cpu implementations of NMS including python implementation (using in FastRCNN) and c++ implementation(torchvision cpu nms operator).


The time cost of Different NMS implementations are shown in the following table,

| function                                             | time cost  |
| ---------------------------------------------------- | ---------- |
| nms_method1                                          | 67.0734 ms |
| nms_method2                                          | 56.8422 ms |
| py_cpu_nms(FastRCNN)                                 | 68.8751 ms |
| cython_cpu_nms(FastRCNN)                             | 61.7921 ms |
| nms_torch(torchvision.ops.nms)                       | 5.5612 ms  |
| nms(our implementation based on torchvision.ops.nms) | 4.1891 ms  |


Finally, this project give a pure cpp implemented efficient NMS which is wrote by using c++ and only depended on numpy, and use `pybind11` to exposes interface to python code.


## Setup

1. prepare the envirionment

```
pybind11==2.10.1
numpy==1.21.6
torchvision==0.12.1
torch==1.12.1
```

2. compile cython and pybind11 implemented version

```bash
bash pybind11_nms/build.sh
bash cython_nms/build.sh
```

3. compare time cost on your machine

```
python test_nms.py
```