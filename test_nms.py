import numpy as np
from python_nms.nms import nms_method1, nms_method2, nms_torch
from python_nms.py_cpu_nms import py_cpu_nms
from cython_nms import cython_cpu_nms
import time
from nms_pybind11 import nms


# generate some bboxes

center_points = np.arange(0, 10) + 0.5
y_center_points, x_center_points = np.meshgrid(center_points, center_points)

x_center_points = x_center_points.reshape(-1, 1)
y_center_points = y_center_points.reshape(-1, 1)
center_points = np.concatenate([y_center_points, x_center_points], axis=-1)

len_center_points = len(center_points)
select_indexes = np.random.randint(0, len_center_points, size=(3500,), dtype=np.int64)
center_points = (center_points[select_indexes, :] / 10 * 640).astype(np.float32)

height_list = np.random.randint(1, 100, size=(3500,), dtype=np.int64)
width_list = np.random.randint(1, 100, size=(3500,), dtype=np.int64)

x1 = np.maximum(center_points[:, 1] - width_list / 2, 0)
x2 = np.maximum(center_points[:, 0] - width_list / 2, 0)
y1 = np.maximum(center_points[:, 1] - height_list / 2, 0)
y2 = np.maximum(center_points[:, 0] - height_list / 2, 0)
points = np.concatenate([x1[..., None], y1[..., None], x2[..., None], y2[..., None]], axis=-1).astype(np.float32)
scores = np.random.rand(3500).astype(np.float32)


# calc function time cost
def calc_function_time_cost(func, *args, **kw_args):
    
    for _ in range(5):
        func(*args, **kw_args)

    
    start_time = time.perf_counter()
    for _ in range(20):
        func(*args, **kw_args)
    timespan = (time.perf_counter() - start_time) * 1000 / 20
    print(f"{func.__name__} cost {timespan:.04f} ms")   
    return func(*args, **kw_args) 


calc_function_time_cost(nms_method1, points, scores, 0.5)
calc_function_time_cost(nms_method2, points, scores, 0.5)
calc_function_time_cost(py_cpu_nms, points, scores, 0.5)
calc_function_time_cost(cython_cpu_nms, points, scores, 0.5)
nms_torch_ret = calc_function_time_cost(nms_torch, points, scores, 0.5)
nms_pybind11_ret = calc_function_time_cost(nms, points, scores, 0.5)

diff = nms_pybind11_ret - nms_torch_ret
print('diff max', np.max(np.abs(diff)))