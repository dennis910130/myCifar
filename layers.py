__author__ = 'chensi'
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


################## dropout technique#############################
def _dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1,p=1-p,size=layer.shape)
    output = layer*T.cast(mask,theano.config.floatX)
    return output
################################################################

def drop_out_layer(rng, input, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1,p=1-p,size=input.shape)
    output = input*T.cast(mask, theano.config.floatX)
    return output

#################Logistic Regression Layer######################
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, rng, n_in, n_out, W='none', b='none', initW=0.01, initB=0,
                 epsW=0.001, epsB=0.002, momW=0.9, momB=0.9, wc=0.004, name='default'):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.name = name
        if W=='none':
            self.W = theano.shared(numpy.asarray(
            rng.normal(loc=0.0, scale=initW, size=(n_in,n_out)),
            dtype=theano.config.floatX),borrow=True)
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b=='none':
            self.b = theano.shared(value=initB*numpy.ones((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        else:
            self.b = b
        self.grad_W = theano.shared(numpy.zeros((n_in,n_out),dtype=theano.config.floatX),name='W_grad',borrow=True)
        self.grad_b = theano.shared(numpy.zeros((n_out,),dtype=theano.config.floatX),name='b_grad',borrow=True)

        self.bestW = numpy.zeros((n_in, n_out))
        self.bestB = numpy.zeros((n_out,))

        self.epsW = epsW
        self.epsB = epsB
        self.momW = momW
        self.momB = momB
        self.wc = wc

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



#############################################################################
##############################Hidden Layer###################################
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W1=None, b1=None,
                 activation='relu', initW=0.01,initB=0,
                 epsW=0.001, epsB=0.002, momW=0.9, momB=0.9, wc=0.004,name='default'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.name = name

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W1 is None:
            W_values = numpy.asarray(rng.normal(
                    loc=0.0,
                    scale=initW,
                    size=(n_in, n_out)), dtype=theano.config.floatX)


            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            W = W1
        if b1 is None:
            b_values = initB*numpy.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            b=b1
        self.grad_W = theano.shared(numpy.zeros((n_in,n_out),dtype=theano.config.floatX),name='W_grad',borrow=True)
        self.grad_b = theano.shared(numpy.zeros((n_out,),dtype=theano.config.floatX),name='b_grad',borrow=True)
        self.W = W
        self.b = b
        self.bestW = numpy.zeros((n_in,n_out))
        self.bestB = numpy.zeros((n_out,))
        self.epsW = epsW
        self.epsB = epsB
        self.momW = momW
        self.momB = momB
        self.wc = wc
        relu = lambda x:x*(x>0)
        lin_output = T.dot(input, self.W) + self.b
        self.output = relu(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]


###########################Hidden Layer with dropout################
class HiddenLayer_dropout(object):
    def __init__(self, rng, input, n_in, n_out, W1=None, b1=None,
                 activation='relu', initW=0.01,initB=0,
                 epsW=0.001, epsB=0.002, momW=0.9, momB=0.9, wc=0.004,drop_out_rates=0.5, name='default'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.name = name

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W1 is None:
            W_values = numpy.asarray(rng.normal(
                    loc=0.0,
                    scale=initW,
                    size=(n_in, n_out)), dtype=theano.config.floatX)


            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            W = W1
        if b1 is None:
            b_values = initB*numpy.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            b=b1
        self.grad_W = theano.shared(numpy.zeros((n_in,n_out),dtype=theano.config.floatX),name='W_grad',borrow=True)
        self.grad_b = theano.shared(numpy.zeros((n_out,),dtype=theano.config.floatX),name='b_grad',borrow=True)
        self.W = W
        self.b = b
        self.bestW = numpy.zeros((n_in,n_out))
        self.bestB = numpy.zeros((n_out,))
        self.epsW = epsW
        self.epsB = epsB
        self.momW = momW
        self.momB = momB
        self.wc = wc
        relu = lambda x:x*(x>0)
        lin_output = T.dot(input, self.W) + self.b
        output1 = relu(lin_output)
        self.output = _dropout_from_layer(rng,output1,p=drop_out_rates)
        # parameters of the model
        self.params = [self.W, self.b]

##############################################################################
#################Traditional ConvLayer########################################
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape,activation='relu', poolsize=(2,2),poolstride=2,pad=0,
                 convstride=1,initW=0.01,initB=0,partial_sum=1,pooling='max',
                 epsW=0.001, epsB=0.002, momW=0.9, momB=0.9, wc=0.004,
                 W1=None,b1=None, name='default'):
        #changes the bc01 to c01b
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[0] == filter_shape[0]
        self.input = input
        self.name = name
        if W1==None:

            self.W = theano.shared(numpy.asarray(
                rng.normal(loc=0.0, scale=initW, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True)
        else:
            self.W = W1
        if b1==None:
        # the bias is a 1D tensor -- one bias per output feature map
            b_values = initB*numpy.ones((filter_shape[3],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b1
        self.grad_W = theano.shared(numpy.zeros(filter_shape,dtype=theano.config.floatX),name='W_grad',borrow=True)
        self.grad_b = theano.shared(numpy.zeros((filter_shape[3],),dtype=theano.config.floatX),name='b_grad',borrow=True)
        self.bestW = numpy.zeros(filter_shape)
        self.bestB = numpy.zeros((filter_shape[3],))
        self.epsW = epsW
        self.epsB = epsB
        self.momW = momW
        self.momB = momB
        self.wc = wc

        print pooling
        print activation

        # convolve input feature maps with filters
        #conv_out = conv.conv2d(input=input, filters=self.W,
                #filter_shape=filter_shape, image_shape=image_shape)
        #input_shuffled = input.dimshuffle(1,2,3,0)
        #filters_shuffled = self.W.dimshuffle(1,2,3,0)
        conv_op = FilterActs(stride=convstride,partial_sum=partial_sum,pad=pad)
        #contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_input = gpu_contiguous(self.input)
        #contiguous_filters = gpu_contiguous(filters_shuffled)
        contiguous_filters = gpu_contiguous(self.W)
        conv_out_shuffled = conv_op(contiguous_input,contiguous_filters)
        if activation=='relu':
            relu = lambda x: x*(x>0)
            conv_out_shuffled = relu(conv_out_shuffled + self.b.dimshuffle(0,'x', 'x', 'x'))
        if activation=='vshape':
            vshape = lambda x:T.abs_(x)
            conv_out_shuffled = vshape(conv_out_shuffled + self.b.dimshuffle(0,'x', 'x', 'x'))
        if activation == 'mrelu':
            mrelu = lambda x: -x*(x<0)
            conv_out_shuffled = mrelu(conv_out_shuffled + self.b.dimshuffle(0,'x', 'x', 'x'))
        if pooling == 'max':
            pool_op = MaxPool(ds=poolsize[0],stride=poolstride)
            #pooled_out_shuffled = pool_op(conv_out_shuffled)
            self.output = pool_op(conv_out_shuffled)

        else:

            side_length = image_shape[1]+2*pad+1-filter_shape[1]
            self.output = mean_pool_c01b(conv_out_shuffled,pool_shape=poolsize,
                                    pool_stride=(poolstride,poolstride),image_shape=(side_length,side_length))
        #pooled_out = pooled_out_shuffled.dimshuffle(3,0,1,2)



        # downsample each feature map individually, using maxpooling
        #pooled_out = downsample.max_pool_2d(input=conv_out,
         #                                   ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]



############################################################################
##################Saliency Conv Layer######################################

class SaliencyConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2),poolstride=2,pad=0,
                 convstride=1,initW=0.01,initB=0,partial_sum=1,pooling='max',
                 epsW=0.001, epsB=0.002, momW=0.9, momB=0.9, wc=0.004,
                 epsG=0.001, epsBeta=0.001, epsT=0.001,
                 momG=0.9,momBeta=0.9,momT=0.9,
                 W1=None,b1=None,T1=None,Gamma1=None,name='default'):
        #changes the bc01 to c01b
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        self.name = name
        assert image_shape[0] == filter_shape[0]
        self.input = input
        if W1==None:

            self.W = theano.shared(numpy.asarray(
                rng.normal(loc=0.0, scale=initW, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True)
        else:
            self.W = W1
        if b1==None:
        # the bias is a 1D tensor -- one bias per output feature map
            b_values = initB*numpy.ones((filter_shape[3],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b1
        self.grad_W = theano.shared(numpy.zeros(filter_shape,dtype=theano.config.floatX),name='W_grad',borrow=True)
        self.grad_b = theano.shared(numpy.zeros((filter_shape[3],),dtype=theano.config.floatX),name='b_grad',borrow=True)
        self.grad_G = theano.shared(numpy.zeros((filter_shape[3],),dtype=theano.config.floatX),name='g_grad',borrow=True)
        self.grad_Beta = theano.shared(numpy.zeros((filter_shape[3],),dtype=theano.config.floatX),name='be_grad',borrow=True)
        self.grad_T = theano.shared(numpy.zeros((filter_shape[3],),dtype=theano.config.floatX),name='T_grad',borrow=True)
        self.bestW = numpy.zeros(filter_shape)
        self.bestB = numpy.zeros((filter_shape[3],))
        self.bestGamma = numpy.zeros((filter_shape[3],))
        self.bestBeta = numpy.zeros((filter_shape[3],))
        self.bestT = numpy.zeros((filter_shape[3],))

        self.epsW = epsW
        self.epsB = epsB
        self.momW = momW
        self.momB = momB
        self.wc = wc
        self.epsG = epsG
        self.epsBeta = epsBeta
        self.epsT = epsT
        self.momG = momG
        self.momT = momT
        self.momBeta = momBeta
        if Gamma1 == None:
            self.Gamma = theano.shared(value=numpy.ones((filter_shape[3],),dtype=theano.config.floatX),name='gamma',borrow=True)
        else:
            self.Gamma = Gamma1
        self.Beta = theano.shared(value=numpy.ones((filter_shape[3],),dtype=theano.config.floatX),name='beta',borrow=True)
        if T1 == None:
            self.T = theano.shared(value=numpy.zeros((filter_shape[3],),dtype=theano.config.floatX),name='T',borrow=True)
        else:
            self.T = T1

        print pooling
        print 'saliency layer'

        # convolve input feature maps with filters
        #conv_out = conv.conv2d(input=input, filters=self.W,
                #filter_shape=filter_shape, image_shape=image_shape)
        #input_shuffled = input.dimshuffle(1,2,3,0)
        #filters_shuffled = self.W.dimshuffle(1,2,3,0)
        conv_op = FilterActs(stride=convstride,partial_sum=partial_sum,pad=pad)
        #contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_input = gpu_contiguous(self.input)
        #contiguous_filters = gpu_contiguous(filters_shuffled)
        contiguous_filters = gpu_contiguous(self.W)
        conv_out_shuffled = conv_op(contiguous_input,contiguous_filters)

        saliency_activation = lambda x:(self.Gamma.dimshuffle(0,'x','x','x')*T.pow(T.abs_(x),1.0)+self.T.dimshuffle(0,'x','x','x'))
        relu = lambda x:x*(x>0)

        conv_out_shuffled = relu(saliency_activation(conv_out_shuffled + self.b.dimshuffle(0,'x', 'x', 'x')))



        if pooling == 'max':
            pool_op = MaxPool(ds=poolsize[0],stride=poolstride)
            #pooled_out_shuffled = pool_op(conv_out_shuffled)
            self.output = pool_op(conv_out_shuffled)
        else:

            side_length = image_shape[1]+2*pad+1-filter_shape[1]
            self.output = mean_pool_c01b(conv_out_shuffled,pool_shape=poolsize,
                                    pool_stride=(poolstride,poolstride),image_shape=(side_length,side_length))
        #pooled_out = pooled_out_shuffled.dimshuffle(3,0,1,2)



        # downsample each feature map individually, using maxpooling
        #pooled_out = downsample.max_pool_2d(input=conv_out,
         #                                   ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b,self.Gamma,self.Beta,self.T]
#####################################################################################
#####################fix 1 layer#####################################################
class Fix_1_discsaliency_layer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape,alphas, image_shape,activation='relu', poolsize=(2,2),poolstride=2,pad=0,
                 convstride=1,partial_sum=1,pooling='max',

                 W1=None,b1=None, name='fixed_layer'):
        #changes the bc01 to c01b
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[0] == filter_shape[0]
        self.input = input
        self.name = name

        self.W = W1

        self.b = b1


        print pooling
        print activation

        # convolve input feature maps with filters
        #conv_out = conv.conv2d(input=input, filters=self.W,
                #filter_shape=filter_shape, image_shape=image_shape)
        #input_shuffled = input.dimshuffle(1,2,3,0)
        #filters_shuffled = self.W.dimshuffle(1,2,3,0)
        conv_op = FilterActs(stride=convstride,partial_sum=partial_sum,pad=pad)
        #contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_input = gpu_contiguous(self.input)
        #contiguous_filters = gpu_contiguous(filters_shuffled)
        contiguous_filters = gpu_contiguous(self.W)
        conv_out_shuffled = conv_op(contiguous_input,contiguous_filters)
        conv_out_shuffled = T.abs_(conv_out_shuffled + self.b.dimshuffle(0,'x','x','x'))
        alpha_0 = alphas
        alpha_1 = T.mean(conv_out_shuffled,axis=2)
        alpha_1 = T.mean(alpha_1,axis=1)
        relu = lambda x:x*(x>0)
        for i in range(128):

            for j in range(32):
                thres = T.log(alpha_0[j]/alpha_1[j,i])/(1./alpha_1[j,i]-1./alpha_0[j])
                if alpha_1[j,i]>alpha_0[j]:
                    conv_out_shuffled[i,:,:,j] = relu(conv_out_shuffled[i,:,:,j] - thres)
                else:
                    conv_out_shuffled[i,:,:,j] = relu(thres - conv_out_shuffled[i,:,:,j])







        if pooling == 'max':
            pool_op = MaxPool(ds=poolsize[0],stride=poolstride)
            #pooled_out_shuffled = pool_op(conv_out_shuffled)
            self.output = pool_op(conv_out_shuffled)

        else:

            side_length = image_shape[1]+2*pad+1-filter_shape[1]
            self.output = mean_pool_c01b(conv_out_shuffled,pool_shape=poolsize,
                                    pool_stride=(poolstride,poolstride),image_shape=(side_length,side_length))

        self.params = [self.W, self.b]