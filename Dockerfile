# docker build -t drise .

FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update
RUN apt -y update

RUN apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    vlc \
    iputils-ping

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk

# upgrade python to version 3.8 (IMPT: remove python3-dev and python3-pip if already installed)
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get install -y python3.8-venv python3.8-dev python3-pip
# Set python3.8 as the default python
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    codetiming \
    jupyter

# install dependencies for ScaledYOLOv4 (optional)
RUN pip3 install --no-cache-dir \
    cython==0.29.19 \
    matplotlib==3.3.3 \
    numpy==1.19.4 \
    opencv-python==4.4.0.46 \
    Pillow==8.0.1 \
    pyyaml==5.3.1 \
    scipy==1.5.4 \
    tensorboard==2.4.0 \
    torch==1.7.0 \
    torchvision==0.8.1 \
    tqdm==4.54.1
RUN cd / && \
    git clone https://github.com/JunnYu/mish-cuda && \
    cd mish-cuda && \
    python3 setup.py build install

# install ScaledYOLOv4 (optional)
RUN pip3 install --no-cache-dir gdown
RUN cd / && \
    git clone https://github.com/yhsmiley/ScaledYOLOv4 && \
    cd ScaledYOLOv4 && \
    git checkout 8b21664888c0604e7ba0e9f146af0e7d6f3c4121 && \
    cd scaledyolov4/weights && \
    bash get_weights.sh && \
    cd ../.. && \
    pip3 install . --no-binary=:all:
