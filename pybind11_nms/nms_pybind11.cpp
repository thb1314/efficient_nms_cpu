#include <vector>
#include <algorithm>
#include <exception>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>


namespace py = pybind11;
using namespace py::literals;


// argsort, the results of this function equal to torch.sort(..., stable=True, decending=True)
template<typename T>
std::vector<int32_t> argsort(const T* array, size_t array_size) {
    std::vector<int32_t> array_index(array_size);
    for(int32_t i = 0; i < (int32_t)array_size; ++i)
        array_index[i] = i;
    std::stable_sort(array_index.begin(), array_index.end(),
                     [&array](int32_t pos1, int32_t pos2){return (array[pos1] > array[pos2]);});
    return array_index;
}


template<typename T>
std::vector<int32_t> argsort(const std::vector<T>& array) {
    return argsort(array.data(), array.size());
}


// borrow ideas from https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
template<typename scalar_t>
py::array_t<int32_t> nms(py::array_t<scalar_t>& dets, py::array_t<scalar_t>& scores, float iou_threshold) {
    if(dets.ndim() != 2) {
        throw std::runtime_error("dets should be a 2 dimension numpy.ndarray");
    }
    if(scores.ndim() != 1) {
        throw std::runtime_error("scores should be a 1 dimension numpy.ndarray");
    }

    py::buffer_info dets_info = dets.request();
    int32_t ndets = dets_info.shape[0];


    py::array x1_t = dets[py::make_tuple(py::ellipsis(), 0)];
    py::array y1_t = dets[py::make_tuple(py::ellipsis(), 1)];
    py::array x2_t = dets[py::make_tuple(py::ellipsis(), 2)];
    py::array y2_t = dets[py::make_tuple(py::ellipsis(), 3)];

    py::module_ np = py::module_::import("numpy");
    x1_t = np.attr("ascontiguousarray")(x1_t);
    y1_t = np.attr("ascontiguousarray")(y1_t);
    x2_t = np.attr("ascontiguousarray")(x2_t);
    y2_t = np.attr("ascontiguousarray")(y2_t);
    py::array_t<scalar_t> areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto scores_info = scores.request();
    scalar_t* scores_ptr = static_cast<scalar_t*>(scores_info.ptr);
    std::vector<int32_t> order_t  = argsort(scores_ptr, ndets);

    auto keep_idx = py::array_t<int32_t>(ndets);
    auto keep_idx_info = keep_idx.request();
    int32_t* keep_idx_ptr = static_cast<int32_t*>(keep_idx_info.ptr);

    auto areas_info = areas_t.request();
    auto x1_info = x1_t.request();
    auto y1_info = y1_t.request();
    auto x2_info = x2_t.request();
    auto y2_info = y2_t.request();

    scalar_t* areas_ptr = static_cast<scalar_t*>(areas_info.ptr);
    scalar_t* x1_ptr = static_cast<scalar_t*>(x1_info.ptr);
    scalar_t* y1_ptr = static_cast<scalar_t*>(y1_info.ptr);
    scalar_t* x2_ptr = static_cast<scalar_t*>(x2_info.ptr);
    scalar_t* y2_ptr = static_cast<scalar_t*>(y2_info.ptr);

    std::vector<uint8_t> suppressed_t(ndets, 0);
    uint8_t* suppressed_ptr = suppressed_t.data();
    const int32_t* order_ptr = order_t.data();

    int32_t num_to_keep = 0;

    for(int32_t _i = 0; _i < ndets; ++_i) {
        auto i = order_ptr[_i];
        if(suppressed_ptr[i] == 1) continue;

        keep_idx_ptr[num_to_keep++] = i;
        auto ix1 = x1_ptr[i];
        auto iy1 = y1_ptr[i];
        auto ix2 = x2_ptr[i];
        auto iy2 = y2_ptr[i];
        auto iarea = areas_ptr[i];

        for(int32_t _j = _i + 1; _j < ndets; ++_j) {
            auto j = order_ptr[_j];
            if(suppressed_ptr[j] == 1) continue;

            auto xx1 = std::max(ix1, x1_ptr[j]);
            auto yy1 = std::max(iy1, y1_ptr[j]);
            auto xx2 = std::min(ix2, x2_ptr[j]);
            auto yy2 = std::min(iy2, y2_ptr[j]);

            auto w = xx2 - xx1;
            if(w <= static_cast<scalar_t>(0)) continue;
            auto h = yy2 - yy1;
            if(h <= static_cast<scalar_t>(0)) continue;

            auto inter = w * h;
            auto over = inter / (iarea + areas_ptr[j] - inter);
            if(over > iou_threshold)
                suppressed_ptr[j] = 1;
        }
    }

    auto ret_keep_idx = py::array_t<int32_t>(num_to_keep);
    auto ret_keep_idx_proxy = ret_keep_idx.mutable_unchecked<1>();
    for(int32_t i = 0; i < num_to_keep; ++i)
        ret_keep_idx_proxy[i] = keep_idx_ptr[i];
    return ret_keep_idx;
}


PYBIND11_MODULE(nms_pybind11, m) {
    m.doc() = "Simple NMS";
    m.def("nms", &nms<float>);
    m.def("nms", &nms<double>);
}
