CURDIR=$(dirname $(readlink -f "$0"))
g++ -O3 -Wall -shared --std=c++11 -fPIC $(python3 -m pybind11 --includes) $CURDIR/nms_pybind11.cpp -o $CURDIR/../nms_pybind11$(python3-config --extension-suffix)

