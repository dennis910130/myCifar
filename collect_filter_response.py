__author__ = 'chensi'

import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from layers import LogisticRegression,HiddenLayer,LeNetConvPoolLayer
from myUtils import load_cifar_data2

import cPickle
import time
import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from myUtils import mean_pool_c01b
import theano.printing
import theano.tensor.shared_randomstreams

class_response_path = '/home/chensi/mylocal/sichen/data/class_response/'

def collect_filter_response(batch_size=128):

    params_file = open('current_best_params.pkl','rb')
    params = cPickle.load(params_file)
    params_file.close()
    layer1_W = theano.shared(params[0],borrow=True)
    layer1_b = theano.shared(params[1],borrow=True)


    train_set_x, train_set_y = load_cifar_data2(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'])
    print train_set_y.get_value().shape


    n_train_batches = train_set_x.get_value(borrow=True).shape[0]

    n_train_batches /= batch_size


    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    img_input = x.reshape((batch_size,3,32,32))
    img_input = img_input.dimshuffle(1,2,3,0)
    conv_op = FilterActs(stride=1,partial_sum=4,pad=2)
    contiguous_input = gpu_contiguous(img_input)
    #contiguous_filters = gpu_contiguous(filters_shuffled)
    contiguous_filters = gpu_contiguous(layer1_W)
    conv_out_shuffled = conv_op(contiguous_input,contiguous_filters)
    linear_output = conv_out_shuffled + layer1_b.dimshuffle(0,'x','x','x')
    final_output = linear_output.dimshuffle(3,0,1,2).flatten(3)

    get_linear_output = theano.function(inputs=[index],outputs=final_output,
                            givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})
    whole_feature_output = get_linear_output(0)

    for i in range(1,n_train_batches):
        linear_temp = get_linear_output(i)

        print '%d/%d' % (i,n_train_batches-1)
        whole_feature_output = numpy.concatenate((whole_feature_output,linear_temp),axis=0)
    print whole_feature_output.shape
    #class_1 = [whole_feature_output[i,0,:] for i in range(total) if train_set_y[i]==0]
    #class_1 = numpy.array(class_1)
    for class_i in range(10):
        temp = train_set_y.get_value()[0:49920]==class_i
        classi = whole_feature_output[temp,:,:]
        name = class_response_path + 'class_'+str(class_i)+'.pkl'
        f_out = open(name,'wb')
        cPickle.dump(classi,f_out,protocol=cPickle.HIGHEST_PROTOCOL)
        f_out.close()





if __name__ == '__main__':
    collect_filter_response()

