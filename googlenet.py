import mxnet as mx
import time

from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

dshape = (128, 3, 224, 224)
nclass = 1000

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), 
        name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, 
            kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=conv, act_type='relu', 
            name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactory(data, num_1x1, num_3x3red, num_3x3, 
        num_d5x5red, num_d5x5, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), 
            name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), 
            name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), 
            pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), 
            name=('%s_5x5' % name), suffix='_reduce')
    cd5x5 = ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), 
            pad=(2, 2), name=('%s_5x5' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), 
            pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), 
            name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.concat(*[c1x1, c3x3, cd5x5, cproj], 
            name='ch_concat_%s_chconcat' % name, dim=1)
    return concat

def get_symbol(num_classes = nclass, **kwargs):
    data = mx.sym.Variable("data")
    conv1 = ConvFactory(data, 64, kernel=(7, 7), stride=(2,2), pad=(3, 3), name="conv1")
    pool1 = mx.sym.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max")
    conv2 = ConvFactory(pool1, 64, kernel=(1, 1), stride=(1,1), name="conv2")
    conv3 = ConvFactory(conv2, 192, kernel=(3, 3), stride=(1, 1), pad=(1,1), name="conv3")
    pool3 = mx.sym.Pooling(conv3, kernel=(3, 3), stride=(2, 2), pool_type="max")

    in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name="in3a")
    in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="in3b")
    pool4 = mx.sym.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name="in4a")
    in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="in4b")
    in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="in4c")
    in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="in4d")
    in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="in4e")
    pool5 = mx.sym.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name="in5a")
    in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="in5b")
    pool6 = mx.sym.Pooling(in5b, kernel=(2, 2), pool_type="avg")
    flatten = mx.sym.Flatten(data=pool6)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return data, softmax

def time_mxnet_symbol(n=100):
    dbatch = mx.nd.random.uniform(-1, 1, shape=dshape, 
            dtype='float32', ctx=mx.gpu())
    _, out = get_symbol()
    sym_exec = out.simple_bind(ctx=mx.gpu(), data=dshape)
    grad = mx.nd.ones((128, 10), ctx=mx.gpu())
    tic = time.time()
    for _ in range(n):
        sym_exec.forward(is_train=True, data=dbatch)
        sym_exec.backward()
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / n

def gluon_symbolblock(n=100):
    inputs, sym = get_symbol()
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
    class Inception(nn.HybridBlock):
        def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # path 1
            self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1,
                                       activation='relu')
            # path 2
            self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1,
                                       activation='relu')
            self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1,
                                       activation='relu')
            # path 3
            self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1,
                                       activation='relu')
            self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2,
                                       activation='relu')
            # path 4
            self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1,
                                          strides=1)
            self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1,
                                       activation='relu')
    
        def forward(self, x):
            p1 = self.p1_conv_1(x)
            p2 = self.p2_conv_3(self.p2_conv_1(x))
            p3 = self.p3_conv_5(self.p3_conv_1(x))
            p4 = self.p4_conv_1(self.p4_pool_3(x))
            if isinstance(p1, mx.sym.Symbol):
                out = mx.sym.concat(p1, p2, p3, p4, dim=1)
            else:
                out = nd.concat(p1, p2, p3, p4, dim=1)
            return out
        
    class GoogLeNet(nn.HybridBlock):
        def __init__(self, num_classes, verbose=False, **kwargs):
            super(GoogLeNet, self).__init__(**kwargs)
            self.verbose = verbose
            # add name_scope on the outer most Sequential
            with self.name_scope():
                # block 1
                b1 = nn.HybridSequential()
                b1.add(
                    nn.Conv2D(64, kernel_size=7, strides=2,
                              padding=3, activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )
                # block 2
                b2 = nn.HybridSequential()
                b2.add(
                    nn.Conv2D(64, kernel_size=1),
                    nn.Conv2D(192, kernel_size=3, padding=1),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )

                # block 3
                b3 = nn.HybridSequential()
                b3.add(
                    Inception(64, 96, 128, 16,32, 32),
                    Inception(128, 128, 192, 32, 96, 64),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )

                # block 4
                b4 = nn.HybridSequential()
                b4.add(
                    Inception(192, 96, 208, 16, 48, 64),
                    Inception(160, 112, 224, 24, 64, 64),
                    Inception(128, 128, 256, 24, 64, 64),
                    Inception(112, 144, 288, 32, 64, 64),
                    Inception(256, 160, 320, 32, 128, 128),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )

                # block 5
                b5 = nn.HybridSequential()
                b5.add(
                    Inception(256, 160, 320, 32, 128, 128),
                    Inception(384, 192, 384, 48, 128, 128),
                    nn.AvgPool2D(pool_size=2)
                )
                # block 6
                b6 = nn.HybridSequential()
                b6.add(
                    nn.Flatten(),
                    nn.Dense(num_classes)
                )
                # chain blocks together
                self.net = nn.HybridSequential()
                self.net.add(b1, b2, b3, b4, b5, b6)

        def forward(self, x):
            out = x
            for i, b in enumerate(self.net):
                out = b(out)
            return out    

    net = GoogLeNet(nclass, verbose=True)
    if hybridize:
        net.hybridize()

    net.collect_params().initialize(mx.init.One(), ctx=mx.gpu())
    dbatch = mx.nd.random.uniform(-1, 1, shape=dshape, 
            dtype='float32', ctx=mx.gpu())
    tic = time.time()
    for _ in range(n):
        with autograd.record():
            out = net(dbatch)
        out.backward()
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / n
  
def gluon_block(n=100):
    class Inception(nn.Block):
        def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # path 1
            self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1,
                                       activation='relu')
            # path 2
            self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1,
                                       activation='relu')
            self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1,
                                       activation='relu')
            # path 3
            self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1,
                                       activation='relu')
            self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2,
                                       activation='relu')
            # path 4
            self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1,
                                          strides=1)
            self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1,
                                       activation='relu')
    
        def forward(self, x):
            p1 = self.p1_conv_1(x)
            p2 = self.p2_conv_3(self.p2_conv_1(x))
            p3 = self.p3_conv_5(self.p3_conv_1(x))
            p4 = self.p4_conv_1(self.p4_pool_3(x))
            return nd.concat(p1, p2, p3, p4, dim=1)
        
    class GoogLeNet(nn.Block):
        def __init__(self, num_classes, verbose=False, **kwargs):
            super(GoogLeNet, self).__init__(**kwargs)
            self.verbose = verbose
            # add name_scope on the outer most Sequential
            with self.name_scope():
                # block 1
                b1 = nn.Sequential()
                b1.add(
                    nn.Conv2D(64, kernel_size=7, strides=2,
                              padding=3, activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )
                # block 2
                b2 = nn.Sequential()
                b2.add(
                    nn.Conv2D(64, kernel_size=1),
                    nn.Conv2D(192, kernel_size=3, padding=1),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )

                # block 3
                b3 = nn.Sequential()
                b3.add(
                    Inception(64, 96, 128, 16,32, 32),
                    Inception(128, 128, 192, 32, 96, 64),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )

                # block 4
                b4 = nn.Sequential()
                b4.add(
                    Inception(192, 96, 208, 16, 48, 64),
                    Inception(160, 112, 224, 24, 64, 64),
                    Inception(128, 128, 256, 24, 64, 64),
                    Inception(112, 144, 288, 32, 64, 64),
                    Inception(256, 160, 320, 32, 128, 128),
                    nn.MaxPool2D(pool_size=3, strides=2)
                )

                # block 5
                b5 = nn.Sequential()
                b5.add(
                    Inception(256, 160, 320, 32, 128, 128),
                    Inception(384, 192, 384, 48, 128, 128),
                    nn.AvgPool2D(pool_size=2)
                )
                # block 6
                b6 = nn.Sequential()
                b6.add(
                    nn.Flatten(),
                    nn.Dense(num_classes)
                )
                # chain blocks together
                self.net = nn.Sequential()
                self.net.add(b1, b2, b3, b4, b5, b6)

        def forward(self, x):
            out = x
            for i, b in enumerate(self.net):
                out = b(out)
            return out    
    net = GoogLeNet(nclass, verbose=True)

    net.collect_params().initialize(mx.init.One(), ctx=mx.gpu())
    dbatch = mx.nd.random.uniform(-1, 1, shape=dshape, 
            dtype='float32', ctx=mx.gpu())
    tic = time.time()
    for _ in range(n):
        with autograd.record():
            out = net(dbatch)
        out.backward()
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic) / n

def main(n=100):
    print('gluon Block run time: {}'.format(gluon_block(n)))
    print('gluon SymbolBlock run time: {}'.format(gluon_symbolblock(n)))
    print('gluon HybridBlock hyridized run time: {}'.format(gluon_hybridblock(n)))
    print('gluon HybridBlock unhyridized run time: {}'.format(
        gluon_hybridblock(n, hybridize=False)))
    print('mxnet symbol run time: {}'.format(time_mxnet_symbol(n)))

main()
