FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Hai Liang Wang <hailiang.hl.wang@gmail.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install virtualenv

# create virtualenv
RUN mkdir -p /virtualenv
WORKDIR /virtualenv
RUN virtualenv --no-site-packages -p /usr/bin/python3.5 py3.5
# RUN virtualenv --no-site-packages -p /usr/bin/python2.7 py2.7

# copy data
RUN mkdir -p /deepqa2
WORKDIR /deepqa2
COPY . .

# install python modules
RUN . /virtualenv/py3.5/bin/activate && \
    pip --no-cache-dir install -r requirements.txt

# Install tensorflow
RUN export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl && \
    . /virtualenv/py3.5/bin/activate && \
    pip install $TF_BINARY_URL


# ENTRYPOINT ["source", "/virtualenv/py3.5/bin/activate", "&&", "python"]
CMD []
