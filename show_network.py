__author__ = 'chensi'
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from layers import LogisticRegression,HiddenLayer,LeNetConvPoolLayer,SaliencyConvLayer
from myUtils import load_cifar_data


def cifar_fast_net(batch_size=1,n_epochs=800,test_frequency=13, learning_rate=0.001):

    params_file = open('current_best_params_saliency.pkl','rb')
    params = cPickle.load(params_file)
    params_file.close()
    layer1_W = theano.shared(params[0],borrow=True)
    layer1_b = theano.shared(params[1],borrow=True)
    layer2_W = theano.shared(params[2],borrow=True)
    layer2_b = theano.shared(params[3],borrow=True)
    layer3_W = theano.shared(params[4],borrow=True)
    layer3_b = theano.shared(params[5],borrow=True)
    fc64_W = theano.shared(params[6],borrow=True)
    fc64_b = theano.shared(params[7],borrow=True)
    fc10_W = theano.shared(params[8],borrow=True)
    fc10_b = theano.shared(params[9],borrow=True)
    layer1_T = theano.shared(params[10],borrow=True)
    layer2_T = theano.shared(params[11],borrow=True)
    layer3_T = theano.shared(params[12],borrow=True)
    layer1_Gamma = theano.shared(params[13],borrow=True)
    layer2_Gamma = theano.shared(params[14],borrow=True)
    layer3_Gamma = theano.shared(params[15],borrow=True)


    rng1 = numpy.random.RandomState(23455)
    rng2 = numpy.random.RandomState(12423)
    rng3 = numpy.random.RandomState(23245)
    rng4 = numpy.random.RandomState(12123)
    rng5 = numpy.random.RandomState(25365)
    rng6 = numpy.random.RandomState(15323)
    train_set_x, train_set_y = load_cifar_data(['data_batch_1','data_batch_2','data_batch_3','data_batch_4'])
    valid_set_x, valid_set_y = load_cifar_data(['data_batch_5'],WHICHSET='valid')
    test_set_x, test_set_y = load_cifar_data(['test_batch'],WHICHSET='test')

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    img_input = x.reshape((batch_size,3,32,32))
    img_input = img_input.dimshuffle(1,2,3,0)
####define the layers:
    conv_pool1 = SaliencyConvLayer(rng=rng1,input=img_input,
                                    filter_shape=(3,5,5,32),
                                    image_shape=(3,32,32,batch_size),
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.0001,initB=0,partial_sum=4,
                                    pooling='max',
                                    epsW=0,
                                    epsB=0,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004,
                                    epsG=0,
                                    epsBeta=0,
                                    epsT=0.00,
                                    W1=layer1_W,
                                    b1=layer1_b,
                                    T1=layer1_T,
                                    Gamma1=layer1_Gamma
                                    )

    conv_pool2 = SaliencyConvLayer(rng=rng2,input=conv_pool1.output,
                                    filter_shape=(32,5,5,32),
                                    image_shape=(32,16,16,batch_size),
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.01,initB=0,partial_sum=4,
                                    pooling='average',
                                    epsW=0,
                                    epsB=0.0000,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004,
                                    epsG=0.000,
                                    epsBeta=0.000,
                                    epsT=0.00,
                                    W1=layer2_W,
                                    b1=layer2_b,
                                    T1=layer2_T,
                                    Gamma1=layer2_Gamma
                                    )
    conv_pool3 = SaliencyConvLayer(rng=rng3,input=conv_pool2.output,
                                    filter_shape=(32,5,5,64),
                                    image_shape=(32,8,8,batch_size),
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.01,initB=0,partial_sum=4,
                                    pooling='average',
                                    epsW=0.0000,
                                    epsB=0.0000,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004,
                                    epsG=0.000,
                                    epsBeta=0.000,
                                    epsT=0.001,
                                    W1=layer3_W,
                                    b1=layer3_b,
                                    T1=layer3_T,
                                    Gamma1=layer3_Gamma
                                    )

    layer4_input = conv_pool3.output.dimshuffle(3,0,1,2).flatten(2)
    #fc_64 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0)
    fc_64 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0,
                        epsW=0.0000,
                        epsB=0.0000,
                        momW=0.9,
                        momB=0.9,
                        wc=0.03,
                        W1=fc64_W*0.5,
                        b1=fc64_b)
    fc_10 = LogisticRegression(input=fc_64.output,rng=rng5,n_in=64,n_out=10,initW=0.1,
                               epsW=0.0000,
                                epsB=0.0000,
                                momW=0.9,
                                momB=0.9,
                                wc=0.03,
                                W=fc10_W,
                                b=fc10_b,
                                )

####build the models:

    look_model = theano.function([index], outputs=(img_input,conv_pool1.output,conv_pool2.output,conv_pool3.output,fc_64.output,fc_10.p_y_given_x),
             givens={
                x: train_set_x[index :(index + 1)]})


    temp = look_model(0)
    f = open('show_network.pkl','wb')
    cPickle.dump(temp,f)
    f.close()




if __name__ == '__main__':
    cifar_fast_net()


