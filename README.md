# Dl4j-Benchmarks

Repository to track Dl4j benchmarks in relation to well known frameworks on cpu and gpu and for dl4j on spark.

#### Core Benchmarks

* MLP: using same model structure used for all frameworks
* Lenet: using same model structure used for all frameworks
* Cifar10: comparing Dl4j against best structures from each framework 
* Other: exploring and works in progress for other comparisons

#### Spark Benchmarks

The deeplearning4j-spark-benchmark package contains a number of synthetic benchmarks to test Spark training performance under a variety of situations.

For more details, see the readme [here - TODO]


## Core Packages Comparison
Main packages included for comparison so far...
**Dl4j (v0.4.1)**
* Install: http://deeplearning4j.org/quickstart
* Setup packages: add to pom.xml
* Set GPU: change in pom file under nd4j-backends (native for cpu and cuda-7.5 for gpu) 
* Compile: mvn clean install -P (native or cuda)
* Run: ./shell_script/dl4j_train_*.sh

**Caffe (vr3)**
* Install: http://caffe.berkeleyvision.org/installation.html
* Set GPU: change in solver prototext under solver_mode as either CPU or GPU
* Pass in -gpu # for the number of GPUs to use or all for all
* Run: ./shell_script/caffe_train_*.sh

**Tensorflow(v0.9.0)**
* Install: https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
* Setup packages: pip install \<*filename*>
* cuDNN required for CNN models and cuDNN > 4 requires to compile from source
* Checkout for configuration fixes: https://stackoverflow.com/questions/37663064/cudnn-compile-configuration-in-tensorflow
* Run: ./shell_script/tensorflow_train_*.sh

**Torch (v7)**
* Install: http://torch.ch/docs/getting-started.html 
* Setup packages: luarocks install \<*filename*>
* Utilize cutorch, cunn, cudnn packages for cuda backend integration 
* cuDNN required for CNN models
* Run: ./shell_script/torch_train_*.sh

If multiple GPUs, control how many used by adding 'export CUDA_VISIBLE_DEVICES=' to .bashrc or .bash_profile and setting empty for CPU and 0,1,2,3 for GPUs (0 if just one and 0,1 if just two) 

#### Benchmark System
Running benchmarks on following system setup:
* Ubuntu 14.0.4
* 60GB RAM 
* 32 Intel Xeon E5-2670 CPUs
* 4 Grid GPUs 4GB RAM
* gcc & g++ v4.9
* BLAS: OpenBLAS v1.13 or Cublas v7.5
* cuDNN v5.1.3

In order to run all examples in core, you need to configure a system for all of the platforms. Each platform differs in requirements and be especially aware of software versions that are not supported.

#### Core Prelim Results

Bottom line this data is preliminary, and we are working to confirm performance. TBV is to be verified where we either do not have any trust in the number that we are getting or we haven't finished getting the script to work (e.g. Torch multi-gpus). Consider all numbers hostile with potential to change as we get additional reviews sorted.

**MLP Example**

| Package    | CPU   | GPU   | Multi | Accuracy |
| ---------- |:-----:| -----:| -----:| --------:| 
| Dl4j       | 9m53s | 2m26s | 58s   | ~97.4%   | 
| Caffe      | 4m21s |   14s | 41s   | ~97.4%   |
| Tensorflow | 1m10s |   45s | TBV   | ~98.3%*  |
| Torch      | 5m14s | 1m11s | TBV   | ~97.9%   |

**Lenet Example w/ cuDNN**

| Package    | CPU   | GPU   | Multi | Accuracy |
| ---------: |------:| -----:| -----:| --------:| 
| Dl4j       | 27m3s | 2m55s | 1m5s  | ~99.0%   | 
| Caffe      | 14m29s|   45s | 1m0s  | ~98.8%   |
| Tensorflow | 5m11s | 1m42s | TBV   | ~98.5%   |
| Torch      | 14m24s| 4m52s | TBV   | ~98.1%   |

Note: 
 * Tensorflow required learning rate modification on MLP by 1/10th otherwise accuracy drops to 9%
 - Accuracy is based on CPU & single GPU.

## *TODO*
Help is welcome to improve comparisons. If you know a better way or see a fix that is needed, please submit a pull request. Top of mind next steps that help would be appreciated:

    - Confirm configurations for all frameworks (seeking outside reviews - esp. on Tensorflow MLP)
    - Setup multi-gpu comparison on all frameworks
    - Compare LSTMs, Autoencoders, RBMs where applicable
    - Setup Dl4j AlexNet functionality with multiple GPUs for benchmark
