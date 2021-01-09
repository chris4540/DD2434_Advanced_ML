#!/bin/bash
#
# A script to untar the data tarball
# To avoid tarball bomb
#


rm -rf ./reuters21578
mkdir -p ./reuters21578

cd ./reuters21578
tar -zxvf ../reuters21578.tar.gz
