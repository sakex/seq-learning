FROM ubuntu:latest

RUN apt-get update

# Install other dependencies
RUN apt-get install -y wget curl python3 python3-pip wget libboost-python-dev libboost-dev build-essential zlib1g-dev \
libboost-system-dev libboost-program-options-dev libarmadillo-dev libboost-numpy-dev gcc-9 g++-9

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10

# Install CMAKE
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:george-edison55/cmake-3.x -y
RUN apt-get update -y
RUN apt-get install cmake -y

WORKDIR /seq-learning/
COPY . /seq-learning/

RUN pip install -r requirements.txt

WORKDIR /seq-learning/Debug/
RUN rm -rf *
RUN cmake ..
RUN cmake --build .

WORKDIR /seq-learning/

RUN python3 -m unittest tests.py
