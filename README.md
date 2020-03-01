# seq-learning

# Build original R package (Linux)

### In R:

    install.packages("devtools") # Build tools for Cpp extensions
  
### You might be missing some dependencies, I personnaly had to run:

    sudo apt install libxml2-dev libcurl4-openssl-dev # I had played with openSSL and changed my installation in the past
  
### Back to R:

    install.packages("RcppArmadillo") # Armadillo R bindings
    install.packages("lhs") # Dependency
    install.packages("numDeriv") # Other dependency
  
### The easiest way to build the package is to use RStudio and run the tests

#### Then you can use the IDE of your choice and run:

    devtools::test()

## Performance

It takes `~10min` to run `tests/testthat/test_C_est.R`, it clearly doesn't use all threads, so there might be some optimization there

<img src="https://github.com/sakex/seq-learning/blob/master/images/cpu_activity.png?raw=true" />
