CURDIR=$(dirname $(readlink -f "$0"))
python $CURDIR/setup.py build
mv $CURDIR/../build/*/*.so $CURDIR/../
rm $CURDIR/../build -rf
rm $CURDIR/cython_nms.c -f
rm $CURDIR/cython_nms.html -f

