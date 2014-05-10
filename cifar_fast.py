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


def cifar_fast_net(batch_size=128,n_epochs=300,test_frequency=13, learning_rate=0.001):

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
    conv_pool1 = LeNetConvPoolLayer(rng=rng1,input=img_input,
                                    filter_shape=(3,5,5,32),
                                    image_shape=(3,32,32,batch_size),
                                    activation='vshape',
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.0001,initB=0,partial_sum=4,
                                    pooling='max',
                                    epsW=0.001,
                                    epsB=0.002,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004
                                    )

    conv_pool2 = LeNetConvPoolLayer(rng=rng2,input=conv_pool1.output,
                                    filter_shape=(32,5,5,32),
                                    image_shape=(32,16,16,batch_size),
                                    activation='vshape',
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.01,initB=0,partial_sum=4,
                                    pooling='average',
                                    epsW=0.001,
                                    epsB=0.002,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004)
    conv_pool3 = LeNetConvPoolLayer(rng=rng3,input=conv_pool2.output,
                                    filter_shape=(32,5,5,64),
                                    image_shape=(32,8,8,batch_size),
                                    activation='vshape',
                                    poolsize=(3,3),poolstride=2,pad=2,
                                    convstride=1,initW=0.01,initB=0,partial_sum=4,
                                    pooling='average',
                                    epsW=0.001,
                                    epsB=0.002,
                                    momW=0.9,
                                    momB=0.9,
                                    wc=0.004)

    layer4_input = conv_pool3.output.dimshuffle(3,0,1,2).flatten(2)
    #fc_64 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0)
    fc_64 = HiddenLayer(rng=rng4,input=layer4_input,n_in=64*4*4,n_out=64,initW=0.1,initB=0,
                        epsW=0.001,
                        epsB=0.002,
                        momW=0.9,
                        momB=0.9,
                        wc=0.03)
    fc_10 = LogisticRegression(input=fc_64.output,rng=rng5,n_in=64,n_out=10,initW=0.1,
                               epsW=0.001,
                                epsB=0.002,
                                momW=0.9,
                                momB=0.9,
                                wc=0.03)
####build the models:
    cost = fc_10.negative_log_likelihood(y)
    test_model = theano.function([index], fc_10.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], fc_10.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    Ws = [conv_pool1.W, conv_pool2.W, conv_pool3.W, fc_64.W, fc_10.W]
    pgradWs = [conv_pool1.grad_W, conv_pool2.grad_W, conv_pool3.grad_W, fc_64.grad_W, fc_10.grad_W]

    bs = [conv_pool1.b, conv_pool2.b, conv_pool3.b, fc_64.b, fc_10.b]
    pgradbs = [conv_pool1.grad_b, conv_pool2.grad_b, conv_pool3.grad_b, fc_64.grad_b, fc_10.grad_b]

    momWs = [conv_pool1.momW, conv_pool2.momW, conv_pool3.momW, fc_64.momW, fc_10.momW]
    momBs = [conv_pool1.momB, conv_pool2.momB, conv_pool3.momB, fc_64.momB, fc_10.momB]
    wcs = [conv_pool1.wc, conv_pool2.wc, conv_pool3.wc, fc_64.wc, fc_10.wc]
    epsWs = [conv_pool1.epsW, conv_pool2.epsW, conv_pool3.epsW, fc_64.epsW, fc_10.epsW]
    epsBs = [conv_pool1.epsB, conv_pool2.epsB, conv_pool3.epsB, fc_64.epsB, fc_10.epsB]

    gradWs = T.grad(cost, Ws)
    gradbs = T.grad(cost, bs)
    updates = []
    for W_i, gradW_i, momW_i, wc_i, epsW_i, pgW_i in zip(Ws,gradWs,momWs,wcs, epsWs,pgradWs):
        grad_i = - epsW_i*gradW_i - wc_i*epsW_i*W_i + momW_i*pgW_i
        updates.append((W_i, W_i+grad_i))
        updates.append((pgW_i,grad_i))

    for b_i, gradb_i, momB_i, epsB_i, pgB_i in zip(bs,gradbs,momBs, epsBs,pgradbs):
        grad_i = - epsB_i*gradb_i + momB_i*pgB_i
        updates.append((b_i, b_i+grad_i))
        updates.append((pgB_i,grad_i))







    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #below is the code for reduce learning_rate
        ###########################################

        if epoch == 50:
            epsWs = [k/10.0 for k in epsWs]
            epsBs = [k/10.0 for k in epsBs]
            print 'reduce eps by a factor of 10'
            updates = []
            for W_i, gradW_i, momW_i, wc_i, epsW_i, pgW_i in zip(Ws,gradWs,momWs,wcs, epsWs,pgradWs):
                grad_i = - epsW_i*gradW_i - wc_i*epsW_i*W_i + momW_i*pgW_i
                updates.append((W_i, W_i+grad_i))
                updates.append((pgW_i,grad_i))

            for b_i, gradb_i, momB_i, epsB_i, pgB_i in zip(bs,gradbs,momBs, epsBs,pgradbs):
                grad_i = - epsB_i*gradb_i + momB_i*pgB_i
                updates.append((b_i, b_i+grad_i))
                updates.append((pgB_i,grad_i))
            train_model = theano.function([index], cost, updates=updates,
              givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})

        ##############################################
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    conv_pool1.bestW = conv_pool1.W.get_value().copy()
                    conv_pool1.bestB = conv_pool1.b.get_value().copy()
                    conv_pool2.bestW = conv_pool2.W.get_value().copy()
                    conv_pool2.bestB = conv_pool2.b.get_value().copy()
                    conv_pool3.bestW = conv_pool3.W.get_value().copy()
                    conv_pool3.bestB = conv_pool3.b.get_value().copy()
                    fc_64.bestW = fc_64.W.get_value().copy()
                    fc_64.bestB = fc_64.b.get_value().copy()
                    fc_10.bestW = fc_10.W.get_value().copy()
                    fc_10.bestB = fc_10.b.get_value().copy()

                    ##saving current best
                    print 'saving current best params..'
                    current_params = (conv_pool1.bestW,conv_pool1.bestB,conv_pool2.bestW,
                    conv_pool2.bestB,conv_pool3.bestW,conv_pool3.bestB,fc_64.bestW,fc_64.bestB,
                    fc_10.bestW,fc_10.bestB,momWs,momBs,epsWs,epsBs,wcs)
                    outfile = file('current_best_params.pkl','wb')
                    cPickle.dump(current_params,outfile)
                    outfile.close()


                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    cifar_fast_net()

