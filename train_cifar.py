__author__ = 'chensi'
import cPickle
import gzip
import os
import sys
import time

import numpy
import theano
import theano.tensor as T
from layers import LogisticRegression,HiddenLayer,LeNetConvPoolLayer
from myUtils import load_cifar_data
import optparse

def train_cifar(learning_rate_base=0.01,batch_size=128,n_epochs=100,test_frequency=1300, check_point_frequency=5000,show_progress_frequency=100):
    check_point_path = os.environ['CHECK_POINT_PATH']
    parser = optparse.OptionParser()
    parser.add_option("-f",dest="filename", default='None')

    (options, args) = parser.parse_args()

    #defining the rngs
    rng1 = numpy.random.RandomState(23455)
    rng2 = numpy.random.RandomState(12423)
    rng3 = numpy.random.RandomState(23245)
    rng4 = numpy.random.RandomState(12123)
    rng5 = numpy.random.RandomState(25365)


    train_set_x, train_set_y = load_cifar_data(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'],UNIT_STD=0)
    test_set_x, test_set_y = load_cifar_data(['test_batch'],WHICHSET='test',UNIT_STD=0)

    n_training_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    n_training_batches /= batch_size
    n_test_batches /= batch_size

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    img_input = x.reshape((batch_size,3,32,32)) #bc01
    img_input = img_input.dimshuffle(1,2,3,0) #c01b

    #####################
    #defining the layers#
    #####################

    if options.filename == 'None':
        print 'start new training...'
        print 'building model...'
        conv_pool1 = LeNetConvPoolLayer(rng=rng1,input=img_input,
                                    filter_shape=(3,5,5,32),
                                    image_shape=(3,32,32,batch_size),
                                    activation='relu',
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.0001,initB=0,partial_sum=4,
                                    pooling='max',
                                    epsW=0.001,
                                    epsB=0.002,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004,
                                    name='conv1'
                                    )

        conv_pool2 = LeNetConvPoolLayer(rng=rng2,input=conv_pool1.output,
                                        filter_shape=(32,5,5,32),
                                        image_shape=(32,16,16,batch_size),
                                        activation='relu',
                                        poolsize=(3,3),poolstride=2,pad=2,
                                        convstride=1,initW=0.01,initB=0,partial_sum=4,
                                        pooling='average',
                                        epsW=0.001,
                                        epsB=0.002,
                                        momW=0.9,
                                        momB=0.9,
                                        wc=0.004,
                                        name='conv2')
        conv_pool3 = LeNetConvPoolLayer(rng=rng3,input=conv_pool2.output,
                                        filter_shape=(32,5,5,64),
                                        image_shape=(32,8,8,batch_size),
                                        activation='relu',
                                        poolsize=(3,3),poolstride=2,pad=2,
                                        convstride=1,initW=0.01,initB=0,partial_sum=4,
                                        pooling='average',
                                        epsW=0.001,
                                        epsB=0.002,
                                        momW=0.9,
                                        momB=0.9,
                                        wc=0.004,
                                        name='conv3')

        layer4_input = conv_pool3.output.dimshuffle(3,0,1,2).flatten(2)
        #fc_64 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0)
        fc_1 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0,
                            epsW=0.001,
                            epsB=0.002,
                            momW=0.9,
                            momB=0.9,
                            wc=0.03,
                            name='fc1')
        fc_2 = LogisticRegression(input=fc_1.output,rng=rng5,n_in=64,n_out=10,initW=0.1,
                                   epsW=0.001,
                                    epsB=0.002,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.03,
                                    name='fc2')
    else:
        print 'resume training %s...' % options.filename

        params_file = open(check_point_path+options.filename,'rb')
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
        print 'building model...'

        conv_pool1 = LeNetConvPoolLayer(rng=rng1,input=img_input,
                                    filter_shape=(3,5,5,32),
                                    image_shape=(3,32,32,batch_size),
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.0001,initB=0,partial_sum=4,
                                    pooling='max',
                                    epsW=0.001,
                                    epsB=0.001,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004,
                                    name='conv1',
                                    W1=layer1_W,
                                    b1=layer1_b
                                    )

        conv_pool2 = LeNetConvPoolLayer(rng=rng2,input=conv_pool1.output,
                                        filter_shape=(32,5,5,32),
                                        image_shape=(32,16,16,batch_size),
                                        poolsize=(3,3),poolstride=2,pad=2,
                                        convstride=1,initW=0.01,initB=0,partial_sum=4,
                                        pooling='average',
                                        epsW=0.001,
                                        epsB=0.001,
                                        momW=0.9,
                                        momB=0.9,
                                        wc=0.004,
                                        name='conv2',
                                        W1=layer2_W,
                                        b1=layer2_b
                                        )
        conv_pool3 = LeNetConvPoolLayer(rng=rng3,input=conv_pool2.output,
                                        filter_shape=(32,5,5,64),
                                        image_shape=(32,8,8,batch_size),
                                        poolsize=(3,3),poolstride=2,pad=2,
                                        convstride=1,initW=0.01,initB=0,partial_sum=4,
                                        pooling='average',
                                        epsW=0.001,
                                        epsB=0.001,
                                        momW=0.9,
                                        momB=0.9,
                                        wc=0.004,
                                        name='conv3',
                                        W1=layer3_W,
                                        b1=layer3_b
                                        )

        layer4_input = conv_pool3.output.dimshuffle(3,0,1,2).flatten(2)
        #fc_64 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0)
        fc_1 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0,
                            epsW=0.001,
                            epsB=0.001,
                            momW=0.9,
                            momB=0.9,
                            wc=0.03,
                            W1=fc64_W,
                            b1=fc64_b,
                            name='fc1')
        fc_2 = LogisticRegression(input=fc_1.output,rng=rng5,n_in=64,n_out=10,initW=0.1,
                                   epsW=0.001,
                                    epsB=0.001,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.03,
                                    W=fc10_W,
                                    b=fc10_b,
                                    name='fc2'
                                    )
    all_layers = [conv_pool1,conv_pool2,conv_pool3,fc_1,fc_2]
#######test model
    cost = fc_2.negative_log_likelihood(y)




    test_model = theano.function(inputs=[index], outputs=fc_2.errors(y),
                                 givens={
                                     x:test_set_x[index*batch_size: (index+1)*batch_size],
                                     y:test_set_y[index*batch_size: (index+1)*batch_size]
                                 })
########train model
    Ws = []
    pgradWs = []

    bs = []
    pgradbs = []

    momWs = []
    mombs = []

    epsWs = []
    epsbs = []
    wcs = []

    for i in range(len(all_layers)):
        Ws.append(all_layers[i].W)
        pgradWs.append(all_layers[i].grad_W)
        bs.append(all_layers[i].b)
        pgradbs.append(all_layers[i].grad_b)
        momWs.append(all_layers[i].momW)
        mombs.append(all_layers[i].momB)
        epsWs.append(all_layers[i].epsW)
        epsbs.append(all_layers[i].epsB)
        wcs.append(all_layers[i].wc)

    gradWs = T.grad(cost, Ws)
    gradbs = T.grad(cost, bs)
    updates = []
    for W_i, gradW_i, momW_i, wc_i, epsW_i, pgW_i in zip(Ws, gradWs, momWs, wcs, epsWs, pgradWs):
        epsW_i *= learning_rate_base
        grad_i = - epsW_i*gradW_i - wc_i*epsW_i*W_i + momW_i*pgW_i

        updates.append((W_i, W_i+grad_i))
        updates.append((pgW_i, grad_i))

    for b_i, gradb_i, momb_i, epsb_i, pgb_i in zip(bs, gradbs, mombs, epsbs, pgradbs):
        grad_i = - epsb_i*gradb_i + momb_i*pgb_i
        updates.append((b_i, b_i+grad_i))
        updates.append((pgb_i,grad_i))

    train_model = theano.function(inputs=[index],outputs=[cost,fc_2.errors(y)],updates=updates,
                                  givens={
                                      x: train_set_x[index*batch_size:(index+1)*batch_size],
                                      y: train_set_y[index*batch_size:(index+1)*batch_size]
                                  })

    #############
    #train model#
    #############
    print 'training...'


    best_validation_loss = numpy.inf
    best_epoch = 0

    epoch = 0

    pweights = []
    pbias = []
    for i in range(len(all_layers)):
        pweights.append(numpy.mean(numpy.abs(all_layers[i].W.get_value()[0,:])))
        pbias.append(numpy.mean(numpy.abs(all_layers[i].b.get_value())))
    time_start = time.time()
    start_time = time.time()
    while(epoch<n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_training_batches):

            iter = (epoch-1)*n_training_batches + minibatch_index
            train_out = train_model(minibatch_index)
            if iter % show_progress_frequency == 0:
                time_end = time.time()
                print 'epoch: %d, batch_num: %d, cost: %f, training_error: %f, (%f seconds)' % (epoch, minibatch_index, train_out[0], train_out[1], time_end-time_start)
                time_start = time.time()

            if (iter+1) % test_frequency == 0:
                time1 = time.time()
                test_losses = [test_model(i) for i in range(n_test_batches)]
                this_test_loss = numpy.mean(test_losses)
                print '=====================testing output==========================='
                print 'epoch: %d, batch_num: %d, test_error: %f ' % (epoch, minibatch_index, this_test_loss*100.)
                for i in range(len(all_layers)):
                    weights = numpy.mean(numpy.abs(all_layers[i].W.get_value()[0,:]))
                    bias = numpy.mean(numpy.abs(all_layers[i].b.get_value()))

                    print 'Layer: %s, weights[0]:%e [%e]' % (all_layers[i].name, weights*1.00, weights-pweights[i])
                    print 'Layer: %s,bias: %e[%e]' % (all_layers[i].name, bias*1.00, bias-pbias[i])
                    pweights[i] = weights
                    pbias[i] = bias
                if this_test_loss < best_validation_loss:
                    best_epoch = epoch
                    best_validation_loss = this_test_loss
                    best_params = []
                    for i in range(len(all_layers)):
                        best_params.append(all_layers[i].W.get_value().copy())
                        best_params.append(all_layers[i].b.get_value().copy())
                    outfile_name = check_point_path+'current_best_params.pkl'
                    outfile = open(outfile_name,'wb')
                    cPickle.dump(best_params,outfile)
                    outfile.close()
                    print 'saved best params to %s' % outfile_name
                time2 = time.time()
                print '==================================================(%f seconds)' % (time2-time1)
            if (iter+1) % check_point_frequency == 0:
                print '~~~~~~~~~~~~~~~~~~saving check_point~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                time1 = time.time()
                current_params = []
                for i in range(len(all_layers)):
                    current_params.append(all_layers[i].W.get_value().copy())
                    current_params.append(all_layers[i].b.get_value().copy())
                outfile_name = check_point_path + 'current_params_' + str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) \
                + '_' + str(time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec)+'.pkl'
                outfile = open(outfile_name,'wb')
                cPickle.dump(current_params,outfile)
                outfile.close()
                print 'saved check_point to %s' % outfile_name
                time2 = time.time()
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(%f seconds)' % (time2-time1)

    end_time = time.time()
    print 'Best test score is %f at epoch %d. Total time:%f hour' % (best_validation_loss * 100., best_epoch, (end_time-start_time)/3600.)

if __name__ == '__main__':
    train_cifar()


















