# Benchmarks of MXNet and Gluon
This is a performance comparison of native MXNet and Gluon

## Introduction

[MXNet](https://mxnet.incubator.apache.org) is a deep learning framework supported by Amazon. I have had experience with MXNet in the year of 2014. At that time, the documentation of MXNet was far from satisfactory, and the amount of operators supported was not small. However, the performance of MXNet in terms of training speed and memory usage outperformed almost all others at that time. (I am talking about you, tensorflow 0.1 :-P)

Recently, MXNet introduced [Gluon](http://gluon.mxnet.io/index.html), which offers high-level abstractions for predefined layers, loss functions, and optimizers. 

I am trying to adapt MXNet / Gluon for my next project. But I want to persue the best speed and memory efficiency, therefore I am looking for the answer that whether I should stick with native mxnet or try the new gluon
## Gluon

 There a three base block classes in Gluon, `Block`, `HybridBlock`, and `SymbolBlock`. `SymbolBlock` seems to provide a wrap outside the original mxnet `symbol` API. `Block` is the new imperative programming API, while `HybridBlock` provide more flexibility: it is similar to the `Block`, but can be `hybridize()` to make a symbolic computational graph, which provides a better performance.
 
 
## Comparison

A natural questions is, what is the overheading of the wrap outside MXNet? I am going to compare the performance of native mxnet, gluon SymbolBlock, hybridized HybrideBlock, and Block on diffferent network architectures.

The code was in alexnet.py. forward and backward were repeated 100 times for average.

The hardware and software platforms are:

macOS 10.13, CUDA 9.0 CuDNN 7.0

Titan Xp, i7-4790K, 32G

### AlexNet

|Framework      | Time (ms)         | 
| ------------- |:-------------:|
| native mxnet      | 41.2 | 
| gluon Block  | 36.9     | 
| gluon SymbolBlock     | 40.8      |  
| gluon HybridBlock  | 36.8     | 
| gluon HybridBlock (hybridized)    | 36.8     |  

 
It seems the network structure is too simple (sometime naive) to show the difference. I am going to test more complex structures.
<<<<<<< HEAD

### GoogLeNet

|Framework      | Time (ms)         | 
| ------------- |:-------------:|
| native mxnet      | 236.6 | 
| gluon SymbolBlock     | 255.1     |  
| gluon HybridBlock  | 270.4    | 
| gluon HybridBlock (hybridized)    | 222.7     |  
| gluon Block  |  272.4 | 
 
Interestingly, the hybridized HybridBlock gives the best performance in both cases.

# Conclusion

 The performance lost due to the wrapping of gluon is minimal.
=======
 
