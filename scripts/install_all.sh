#!/bin/bash


mkdir -p /public/roberto.scolaro

rm miniconda.sh

wget 'https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh' -O miniconda.sh

chmod +x miniconda.sh

./miniconda.sh

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.2.0

pip install --no-cache-dir tensorflow \
	matplotlib \
	pandas \
	opencv-python \
	jupyterlab \
	numpy

git clone 'https://github.com/aleju/imgaug'

cd imgaug

pip install --no-cache-dir .
