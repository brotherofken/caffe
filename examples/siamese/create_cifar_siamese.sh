#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.
set -e

EXAMPLES=examples/siamese
DATA=./data/cifar10
DBTYPE=lmdb

echo "Creating leveldb..."

rm -rf ./examples/siamese/cifar10_train_$DBTYPE
rm -rf ./examples/siamese/cifar10_test_$DBTYPE

./build/$EXAMPLES/convert_cifar_siamese_data.bin $DATA $EXAMPLES $DBTYPE

#$EXAMPLES/convert_mnist_siamese_data.bin \
#    $DATA/train-images-idx3-ubyte \
#    $DATA/train-labels-idx1-ubyte \
#    ./examples/siamese/mnist_siamese_train_leveldb
#$EXAMPLES/convert_mnist_siamese_data.bin \
#    $DATA/t10k-images-idx3-ubyte \
#    $DATA/t10k-labels-idx1-ubyte \
#    ./examples/siamese/mnist_siamese_test_leveldb

echo "Done."
