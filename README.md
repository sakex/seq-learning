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
