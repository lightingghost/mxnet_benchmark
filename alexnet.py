import mxnet as mx
import time

from mxnet import nd, autograd
from mxnet import gluon

dshape = (128, 3, 224, 224)

def mxnet_symbol():
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=64)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=192)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), 
            pool_type="max")
    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), 
            pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    # stage 5
    fc2 = mx.symbol.FullyConnected(data=relu6, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=relu7, num_hidden=10)

    return input_data, fc3

def time_mxnet_symbol(n=100):
    dbatch = mx.nd.random.uniform(-1, 1, shape=dshape, 
            dtype='float32', ctx=mx.gpu())
    _, fc3 = mxnet_symbol()
    sym_exec = fc3.simple_bind(ctx=mx.gpu(), data=dshape)
    grad = mx.nd.ones((128, 10), ctx=mx.gpu())
    tic = time.time()
    for _ in range(n):
        sym_exec.forward(is_train=True, data=dbatch)
        sym_exec.backward([grad])
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / n

def gluon_symbolblock(n=100):
    inputs, sym = mxnet_symbol()
    model = gluon.SymbolBlock(sym, inputs)
    model.collect_params().initialize(mx.init.One(), ctx=mx.gpu())
    dbatch = mx.nd.random.uniform(-1, 1, shape=dshape, 
            dtype='float32', ctx=mx.gpu())
    tic = time.time()
    for _ in range(n):
        with autograd.record():
            out = model(dbatch)
        out.backward()
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / n

def gluon_hybridblock(n=100, hybridize=True):
    alex_net = gluon.nn.HybridSequential()
    with alex_net.name_scope():
        #  First convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=64, kernel_size=11, 
            strides=(4,4), activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
        #  Second convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, 
            padding=(2, 2), activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))
        # Third convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, 
            padding=(1, 1), activation='relu'))
        # Fourth convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, 
            padding=(1, 1), activation='relu'))
        # Fifth convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, 
            padding=(1, 1), activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
        # Flatten and apply fullly connected layers
        alex_net.add(gluon.nn.Flatten())
        alex_net.add(gluon.nn.Dense(4096, activation="relu"))
        alex_net.add(gluon.nn.Dense(4096, activation="relu"))
        alex_net.add(gluon.nn.Dense(10))
    if hybridize:
        alex_net.hybridize()

    alex_net.collect_params().initialize(mx.init.One(), ctx=mx.gpu())
    dbatch = mx.nd.random.uniform(-1, 1, shape=dshape, 
            dtype='float32', ctx=mx.gpu())
    tic = time.time()
    for _ in range(n):
        with autograd.record():
            out = alex_net(dbatch)
        out.backward()
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / n
  

def gluon_block(n=100):
    alex_net = gluon.nn.Sequential()
    with alex_net.name_scope():
        #  First convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=64, kernel_size=11, 
            strides=(4,4), activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
        #  Second convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, 
            padding=(2, 2), activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))
        # Third convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, 
            padding=(1, 1), activation='relu'))
        # Fourth convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, 
            padding=(1, 1), activation='relu'))
        # Fifth convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, 
            padding=(1, 1), activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
        # Flatten and apply fullly connected layers
        alex_net.add(gluon.nn.Flatten())
        alex_net.add(gluon.nn.Dense(4096, activation="relu"))
        alex_net.add(gluon.nn.Dense(4096, activation="relu"))
        alex_net.add(gluon.nn.Dense(10))

    alex_net.collect_params().initialize(mx.init.One(), ctx=mx.gpu())
    dbatch = mx.nd.random.uniform(-1, 1, shape=dshape, 
            dtype='float32', ctx=mx.gpu())
    tic = time.time()
    for _ in range(n):
        with autograd.record():
            out = alex_net(dbatch)
        out.backward()
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / n
 
def main(n=100):
    print('gluon SymbolBlock run time: {}'.format(gluon_symbolblock(n)))
    print('gluon HybridBlock hyridized run time: {}'.format(gluon_hybridblock(n)))
    print('gluon HybridBlock unhyridized run time: {}'.format(
        gluon_hybridblock(n, hybridize=False)))
    print('gluon Block run time: {}'.format(gluon_block(n)))
    print('mxnet symbol run time: {}'.format(time_mxnet_symbol(n)))

main()
