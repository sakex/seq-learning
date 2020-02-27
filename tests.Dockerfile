FROM ubuntu:latest

RUN apt-get update

# Install CMAKE
RUN wget http://www.cmake.org/files/v3.14/cmake-3.14.4.tar.gz
RUN tar -xvzf cmake-3.14.4.tar.gz
WORKDIR cmake-3.14.4/
RUN ./configure
RUN make -j$(nproc)
RUN make install
RUN update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force

# Install other dependencies
RUN apt-get install -y curl python3 python3-pip wget libboost-python-dev\
libboost-dev build-essential zlib1g-dev libboost-system-dev libboost-program-options-dev

WORKDIR /seq-learning
COPY . /seq-learning

RUN cmake -DCMAKE_TYPE=Release Release
RUN cmake --build Release