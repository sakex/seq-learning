FROM ubuntu:latest

RUN apt-get update

# Install other dependencies
RUN apt-get install -y wget curl python3 python3-pip wget \
libboost-python-dev libboost-dev build-essential zlib1g-dev \
libboost-system-dev libboost-program-options-dev libarmadillo-dev python3-numpy

# Install CMAKE
RUN wget http://www.cmake.org/files/v3.14/cmake-3.14.4.tar.gz
RUN tar -xvzf cmake-3.14.4.tar.gz
WORKDIR cmake-3.14.4/
RUN ./configure
RUN make -j$(nproc)
RUN make install
RUN update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force

WORKDIR /seq-learning/
COPY . /seq-learning/

RUN pip install -r requirements.txt

WORKDIR /seq-learning/Debug/
RUN rm -rf *
RUN cmake ..
RUN cmake --build .

WORKDIR /seq-learning/

RUN python3 -m unittest tests.py
