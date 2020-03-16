# seq-learning

# Install

    git clone https://github.com/sakex/seq-learning.git
    
**Or use SSH:**

    git@github.com:sakex/seq-learning.git
    
**Install dependencies**

    pip3 install -r requirements.txt
    
    sudo apt-get install -y wget curl python3 python3-pip wget \
         libboost-python-dev libboost-dev build-essential zlib1g-dev \
         libboost-system-dev libboost-program-options-dev libarmadillo-dev libboost-numpy-dev
         
# Build

    mkdir Debug
    cmake . Debug
    cmake --build Debug
    
# Run

    python3 examples/egeinspace.py

# Build original R package (Linux)

### In R:

    install.packages("devtools") # Build tools for Cpp extensions
  
### You might be missing some dependencies, I personnaly had to run:

    sudo apt install libxml2-dev libcurl4-openssl-dev # I had played with openSSL and changed my installation in the past
  
### Back to R:

    install.packages("RcppArmadillo") # Armadillo R bindings
    # Didn't work on GCC-9, I had to pull GCC-7 (missing -lgfortran, whatever that is ;) )
    install.packages("lhs") # Dependency
    install.packages("numDeriv") # Other dependency
  
### Install

    R CMD INSTALL activegp_1.0.3.tar.gz

## Performance

It takes `~10min` to run `tests/testthat/test_C_est.R`, it clearly doesn't use all threads, so there might be some optimization there

<img src="https://github.com/sakex/seq-learning/blob/master/images/cpu_activity.png?raw=true" />
