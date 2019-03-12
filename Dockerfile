######################################
# This Dockerfile builds an image that allows you to perform deep reinforcement
# learning with OpenAI Gym hooked up to JSBSim flight simulator (with optional
# visualization with Flight Gear.)  It also contains a VNC server that allows
# you to visualize the training and replay, and the VNC server can be accessed
# via a web browser so no VNC client needs to be installed.
#
# The main contribution of this file is to bring together all the parts and
# pieces from many other projects out there into a single Docker image so you
# can focus on the deep learning rather than figuring out all the gotchas with
# setting up the environment.
#
# The resulting docker image will be huge, and there's no attempt made here to
# optimize it for size.
#
# This image must be run on a system with an NVIDIA CUDA capable GPU.  The
# host machine where it is built and run must have nvidia-docker 2 installed.
# The host machine where this is built and run can be a remote machine such
# as a cloud GPU machine using docker-machine, but this is complex to set up
# and is largely left as an exercise for the reader.  I have run this on
# linux boxes in my basement, my lab, the cloud, and an NVIDIA DGX-1 and it
# has worked in all of these environments.  The host computer must be a linux
# computer, as nvidia-docker does not work on Windows due to Windows limitations.
######################################

# From:
# https://raw.githubusercontent.com/ufoym/deepo/master/docker/Dockerfile.all-py36-cu90
#
# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# tensorflow    latest (pip)
# theano        latest (git)
# keras         latest (pip)
# opencv        3.4.3  (git)
# ==================================================================
#
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN rm -rf /var/lib/apt/lists/* \
       /etc/apt/sources.list.d/cuda.list \
       /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update

######################################
# There are some monster downloads as part of this Docker image.  To make
# things easier for me as I repeatedly build it, I have my docker host
# cache all my apt-get downloads so that the next time they download they
# are fetched from the cache.
######################################

RUN apt-get install -y net-tools inetutils-ping netcat
# If host is running squid-deb-proxy on port 8000, populate /etc/apt/apt.conf.d/30proxy
# By default, squid-deb-proxy 403s unknown sources, so apt shouldn't proxy ppa.launchpad.net
RUN route -n | awk '/^0.0.0.0/ {print $2}' > /tmp/host_ip.txt
RUN echo "HEAD /" | nc `cat /tmp/host_ip.txt` 8000 | grep squid-deb-proxy \
  && (echo "Acquire::http::Proxy \"http://$(cat /tmp/host_ip.txt):8000\";" > /etc/apt/apt.conf.d/30proxy) \
  || echo "No squid-deb-proxy detected on docker host"

#&& (echo "Acquire::http::Proxy::ppa.launchpad.net DIRECT;" >> /etc/apt/apt.conf.d/30proxy) \

RUN ls -l /etc/apt/apt.conf.d/

######################################
# Install all the deep learning tools that I need. (And a couple that I don't
# technically use, but assume I will at some point.)
######################################

ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
ENV GIT_CLONE="git clone --depth 10"


# ==================================================================
# tools
# ------------------------------------------------------------------

RUN    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim


# ==================================================================
# python
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common

RUN echo deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main > /etc/apt/sources.list.d/deadsnakes-ppa-trusty.list

RUN    add-apt-repository ppa:deadsnakes/ppa
RUN    apt-get update
RUN    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev

RUN    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools

RUN    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn \
        matplotlib \
        Cython


# ==================================================================
# tensorflow
# ------------------------------------------------------------------

RUN    $PIP_INSTALL \
        tensorflow-gpu


# ==================================================================
# theano
# ------------------------------------------------------------------

RUN    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libblas-dev


RUN    wget -qO- https://github.com/Theano/libgpuarray/archive/v0.7.6.tar.gz | tar xz -C ~ && \
    cd ~/libgpuarray* && mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j"$(nproc)" install && \
    cd ~/libgpuarray* && \
    python setup.py build && \
    python setup.py install && \

    printf '[global]\nfloatX = float32\ndevice = cuda0\n\n[dnn]\ninclude_path = /usr/local/cuda/targets/x86_64-linux/include\n' > ~/.theanorc && \

    $PIP_INSTALL \
        https://github.com/Theano/Theano/archive/master.zip


# ==================================================================
# keras
# ------------------------------------------------------------------

RUN $PIP_INSTALL \
        h5py \
        keras



# ==================================================================
# opencv
# ------------------------------------------------------------------

RUN    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        && \

    $GIT_CLONE --branch 3.4.3 https://github.com/opencv/opencv ~/opencv && \
    mkdir -p ~/opencv/build && cd ~/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          .. && \
    make -j"$(nproc)" install


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig

#RUN    apt-get clean && \
#    apt-get autoremove && \
#    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 6006

######################################
# Install flightgear
######################################

RUN apt install -y sudo
RUN sudo add-apt-repository -y ppa:saiarcot895/flightgear
RUN sudo apt-get update
# Install flightgear -- note that this takes *forever* due to large downloads
# required.  We want this to be as high as possible in the Dockerfile so that
# we don't accidentally repeatedly have to download it.
RUN sudo apt-get install -y flightgear

RUN sudo update-alternatives --install /usr/bin/fgfs fgfs /usr/games/fgfs 1

######################################
# Install a VNC server and a web browser based VNC server so we can
# remote into our container and view progress while it trains.  This
# method was taken from the consol VNC image.  It has some oddness
# related to definition of $HOME env var that needs cleaned up, but
# it works well enough for now.
######################################

## Connection ports for controlling the UI:
# VNC port:5901
# noVNC webport, connect via http://IP:6901/?password=vncpassword
ENV DISPLAY=:1 \
    VNC_PORT=5901 \
    NO_VNC_PORT=6901
EXPOSE $VNC_PORT $NO_VNC_PORT

### Envrionment config
ENV HOME=/headless \
    TERM=xterm \
    STARTUPDIR=/dockerstartup \
    INST_SCRIPTS=/headless/install \
    NO_VNC_HOME=/headless/noVNC \
    DEBIAN_FRONTEND=noninteractive \
    VNC_COL_DEPTH=24 \
    VNC_RESOLUTION=1280x1024 \
    VNC_PW=vncpassword \
    VNC_VIEW_ONLY=false
WORKDIR $HOME

### Add all install scripts for further steps
ADD ./src/common/install/ $INST_SCRIPTS/
ADD ./src/ubuntu/install/ $INST_SCRIPTS/
RUN find $INST_SCRIPTS -name '*.sh' -exec chmod a+x {} +

### Install some common tools
RUN $INST_SCRIPTS/tools.sh
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

### Install custom fonts
RUN $INST_SCRIPTS/install_custom_fonts.sh

### Install xvnc-server & noVNC - HTML5 based VNC viewer
RUN $INST_SCRIPTS/tigervnc.sh
RUN $INST_SCRIPTS/no_vnc.sh

### Install firefox and chrome browser
RUN $INST_SCRIPTS/firefox.sh
RUN $INST_SCRIPTS/chrome.sh

### Install xfce UI
RUN $INST_SCRIPTS/xfce_ui.sh
ADD ./src/common/xfce/ $HOME/

### configure startup
RUN $INST_SCRIPTS/libnss_wrapper.sh
ADD ./src/common/scripts $STARTUPDIR
RUN $INST_SCRIPTS/set_user_permission.sh $STARTUPDIR $HOME


# Change wallpaper
COPY xfce4-desktop.xml /headless/.config/xfce4/xfconf/xfce-perchannel-xml/

# For some reason the websockify version they use doesn't have websocketproxy
# module and errors out.  Delete it and install whatever websockify is latest
RUN rm -rf /headless/noVNC/utils/websockify
RUN pip install websockify


######################################
# Add a user to our image.  This isn't technically necessary for a docker
# image, but it's probably easier for people to do things in a /home/jsbsim
# directory than it would be to run things from /.  It should be possilbe
# to remove this user, modify all the paths below here, and still get a
# working image that runs as root from the root directory /.
######################################

## Install as root
USER 0

RUN apt-get update && apt-get install -y sudo

# Create a user called "jsbsim", give it sudo access and remove the requirement for a password:
RUN useradd --create-home --shell /bin/bash --no-log-init --groups sudo jsbsim
RUN sudo bash -c 'echo "jsbsim ALL=(ALL:ALL) NOPASSWD: ALL" | (EDITOR="tee -a" visudo)'

######################################
# Install things here that won't vary depending on our code changes
######################################

RUN apt install -y python3-pip
RUN apt install -y git

RUN apt install -y cmake
RUN apt install -y cython3

RUN pip3 install numpy pandas scipy

RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
RUN sudo update-alternatives --install /usr/bin/cython cython /usr/bin/cython3   1
RUN sudo update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3      1

RUN sudo apt-get update && sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev

RUN pip install gym[atari]
RUN pip install joblib

# OpenAI gym (or maybe the atari gym) requires tkinter.  Note that with python3.6
# you must install python3.6-tk.  python3-tk doesn't work.
RUN apt install -y python3.6-tk
RUN pip install mpi4py zmq dill glob2 click progressbar2 seaborn opencv-python tqdm python-utils ipython
RUN apt install -y dos2unix python3-dev swig
RUN apt install -y python-pygame

WORKDIR /home/jsbsim

RUN git clone https://github.com/pybox2d/pybox2d

WORKDIR /home/jsbsim/pybox2d
RUN python setup.py build
RUN sudo python setup.py install

RUN sudo apt-get install libboost-all-dev -y

RUN sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y
RUN sudo apt-get update
RUN sudo apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev -y
RUN sudo apt-get install dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev -y

RUN sudo -E apt-get install libav-tools libsdl2-dev swig cmake -y

######################################
# Now that all the non-varying things are installed that we need, let's bring
# in our code changes.  This will minimize the time it takes to rebuild the
# docker image.
######################################

# Bring in the jsbsim codebase
COPY --chown=jsbsim:jsbsim ./jsbsim-code /home/jsbsim/jsbsim-code

# Build the jsbsim codebase
RUN mkdir -p /home/jsbsim/jsbsim-code/build
WORKDIR /home/jsbsim/jsbsim-code/build
RUN cmake -DINSTALL_PYTHON_MODULE=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -mtune=native" -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -mtune=native" -DCMAKE_BUILD_TYPE=Release ..
RUN make -j4
RUN make install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Bring in the openai baselines
COPY --chown=jsbsim:jsbsim ./baselines /home/jsbsim/baselines

# Install the gym-jsbsim python module from source
WORKDIR /home/jsbsim/
COPY --chown=jsbsim:jsbsim ./gym-jsbsim /home/jsbsim/gym-jsbsim
RUN pip install -e gym-jsbsim

# Install the gym-jsbsim python module from source
WORKDIR /home/jsbsim/
COPY --chown=jsbsim:jsbsim ./stable-baselines /home/jsbsim/stable-baselines
RUN pip install -e stable-baselines

# Bring in rl coach
COPY --chown=jsbsim:jsbsim ./coach /home/jsbsim/coach
RUN pip install -e coach


# Bring in any helpful scripts we have written
COPY --chown=jsbsim:jsbsim ./*.sh /home/jsbsim/
COPY --chown=jsbsim:jsbsim ./*.py /home/jsbsim/

# Fix permissions on any *.sh file
WORKDIR /home/jsbsim/
RUN dos2unix *.sh
RUN chmod a+x *.sh

# The code is set up to save models in this directory.  We can therefore
# use a volume to save model runs over docker restarts.
RUN mkdir /home/jsbsim/model

# JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0

ENV OPENAI_LOGDIR=/home/jsbsim/logs
ENV OPENAI_LOG_FORMAT=stdout,tensorboard

# For some reason libcupti path is not added to LD library path during
# cuda installs?
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Set up our auxiliary command to run tensorboard in the background on port 8008
# Found that this works better when you launch it from a shell from within the
# environment... that way you can kill the tensorboard if you need to and restart
# it.
#ENV AUX_CMD="tensorboard --logdir /home/jsbsim/model/ --port 8008"