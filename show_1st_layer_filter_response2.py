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
from utils import tile_raster_images
import PIL.Image
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

class_response_path = '/home/chensi/mylocal/sichen/data/class_response/filter_response_imgs/'
alphas_path = ''

def collect_filter_response(batch_size=128):
    alpha_file = open(alphas_path,'rb')
    alphas = cPickle.load(alpha_file)
    alpha_file.close()

    params_file = open('current_best_params.pkl','rb')
    params = cPickle.load(params_file)
    params_file.close()
    layer1_W = theano.shared(params[0],borrow=True)
    layer1_b = theano.shared(params[1],borrow=True)


    train_set_x, train_set_y = load_cifar_data2(['data_batch_1'])
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
    final_output = linear_output.dimshuffle(3,0,1,2)
    final_output = final_output.reshape((batch_size,32,1024))

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
        temp = train_set_y.get_value()[0:9984]==class_i

        temp_data = whole_feature_output[temp,:,:]
        temp_data = temp_data[0:100,:,:]
        temp_data = temp_data + 0.0
        #print temp_data.dtype
        for i in range(32):
            x0 = temp_data[:,i,:]
            #alpha_1 = alphas[class_i,i]
            alpha_0 = numpy.mean(alphas[:,i])
            x0 = numpy.abs(x0)
            for j in range(128):
                alpha_1 = numpy.mean(x0[j,:])
                thres = numpy.log(alpha_0/alpha_1)/(1./alpha_1-1./alpha_0)
                if alpha_0<alpha_1:
                    x0[j,:] = x0[j,:]-thres
                    x0[j,:] = x0[j,:]*(x0[j,:]>0)
                else:
                    x0[j,:] = thres - x0[j,:]
                    x0[j,:] = x0[j,:]*(x0[j,:]>0)


            #not_i = numpy.array(range(10))
            #ind = not_i != class_i






            image = PIL.Image.fromarray(tile_raster_images(X=x0,
                                                           img_shape=(32,32),
                                                           tile_shape=(10,10),
                                                           tile_spacing=(1,1)))
            name = 'class_' + str(class_i) + '_filter_' + str(i) + '.png'
            image.save(class_response_path+name)





if __name__ == '__main__':
    collect_filter_response()

