__author__ = 'chensi'
import theano
import cPickle
import numpy
import theano.tensor as T
from utils import tile_raster_images
import PIL.Image

import numpy as np

from theano.gof.op import get_debug_values






def load_cifar_data(dataset, REMOVE_MEAN=1, UNIT_STD=0, WHICHSET='train'):
    #load the data
    print 'loading data..'
    for batch in range(len(dataset)):
        fo = open(dataset[batch],'rb')
        data = cPickle.load(fo)
        if batch == 0:
            feature = data['data']
            label = data['labels']
        else:
            feature = numpy.concatenate((feature,data['data']),axis=0)
            label = numpy.concatenate((label,data['labels']),axis=0)
        fo.close()


    if WHICHSET == 'train':
        print 'training_set..'
        if REMOVE_MEAN == 1:
            mean = numpy.mean(feature,axis=0)
            mean_file = open('mean_file.pkl','wb')
            cPickle.dump(mean,mean_file,protocol=cPickle.HIGHEST_PROTOCOL)
            mean_file.close()
            feature = feature - mean
            print mean
            print 'feature mean = %f' % (numpy.mean(feature))
        if UNIT_STD == 1:
            std = numpy.std(feature,axis=0)
            feature = feature/std
            std_file = open('std_file.pkl','wb')
            cPickle.dump(std,std_file,protocol=cPickle.HIGHEST_PROTOCOL)
            std_file.close()
            print std
            print 'feature std = %f' % (numpy.std(feature))
    else:
        print 'valid or test_set..'
        if REMOVE_MEAN == 1:
            mean_file = open('mean_file.pkl','rb')
            mean = cPickle.load(mean_file)
            feature = feature - mean
            mean_file.close()
            print mean
            print 'feature mean = %f' % (numpy.mean(feature))
        if UNIT_STD == 1:
            std_file = open('std_file.pkl','rb')
            std = cPickle.load(std_file)
            feature = feature/std
            std_file.close()
            print std
            print 'feature std = %f' % (numpy.std(feature))
    shared_feature = theano.shared(numpy.asarray(feature,dtype=theano.config.floatX),borrow=True)
    shared_label = theano.shared(numpy.asarray(label,dtype=theano.config.floatX),borrow=True)

    return shared_feature, T.cast(shared_label,'int32')

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present



    print '... loading data'

    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def lrn_same_map(c01b,size,pow,scale,image_side):
    mx = None
    for c01bv in get_debug_values(c01b):
        assert not np.any(np.isinf(c01bv))
        assert c01bv.shape[1] == image_side
        assert c01bv.shape[2] == image_side
        
    new_side = size-1+image_side


    wide_infinity = T.alloc(0.0,
                        c01b.shape[0],
                        new_side,
                        new_side,
            	c01b.shape[3])
            	
            	
    c01b_pad = T.set_subtensor(wide_infinity[:, 1:1+image_side, 1:1+image_side, :], T.sqr(c01b))
	


    wide_infinity_count = T.alloc(0,  c01b.shape[0], new_side,
                                  new_side,c01b.shape[3])
    c01b_count = T.set_subtensor(wide_infinity_count[:, 1:1+image_side, 1:1+image_side, :], 1)
    for row_within_pool in xrange(size):
        row_stop = image_side + row_within_pool
        for col_within_pool in xrange(size):
            col_stop = image_side + col_within_pool
            cur = c01b_pad[:,
                       row_within_pool:row_stop:1,
                       col_within_pool:col_stop:1,
			            :]

            cur_count = c01b_count[:,
                                   row_within_pool:row_stop:1,
                                   col_within_pool:col_stop:1,
					        :]
            if mx is None:
                mx = cur
                count = cur_count
            else:
                mx = mx + cur
                count = count + cur_count


    mx /= count
    mx = scale*mx
    mx = mx+1
    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))
    new_c01b = c01b/T.pow(mx,pow)
    return new_c01b


def mean_pool_c01b(c01b,pool_shape,pool_stride,image_shape):
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]#24
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr#27

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc#27

    for c01bv in get_debug_values(c01b):
        assert not np.any(np.isinf(c01bv))
        assert c01bv.shape[1] == image_shape[0]
        assert c01bv.shape[2] == image_shape[1]

    wide_infinity = T.alloc(0.0,
                            c01b.shape[0],
                            required_r,
                            required_c,
			    c01b.shape[3])

    name = c01b.name
    if name is None:
        name = 'anon_c01b'
    c01b = T.set_subtensor(wide_infinity[:, 0:r, 0:c, :], c01b)
    c01b.name = 'infinite_padded_' + name

    # Create a 'mask' used to keep count of the number of elements summed for
    # each position
    wide_infinity_count = T.alloc(0,  c01b.shape[0], required_r,
                                  required_c,c01b.shape[3])
    c01b_count = T.set_subtensor(wide_infinity_count[:, 0:r, 0:c, :], 1)

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = c01b[:,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs,
			:]
            cur.name = ('mean_pool_cur_' + c01b.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            cur_count = c01b_count[:,
                                   row_within_pool:row_stop:rs,
                                   col_within_pool:col_stop:cs,
					:]
            if mx is None:
                mx = cur
                count = cur_count
            else:
                mx = mx + cur
                count = count + cur_count
                mx.name = ('mean_pool_mx_' + c01b.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))

    mx /= count
    mx.name = 'mean_pool_c01b('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx

def filter_visualize_color(params):
    a = open(params,'rb')
    temp = cPickle.load(a)
    w1_c01b = temp[0]
    w1_bc01 = numpy.rollaxis(w1_c01b,3,0)
    w1_shape = w1_bc01.shape
    x = w1_bc01.reshape((w1_shape[0],3,w1_shape[2]*w1_shape[3]))
    x0 = x[:,0,:]
    x1 = x[:,1,:]
    x2 = x[:,2,:]
    image = PIL.Image.fromarray(tile_raster_images(X=(x0,x1,x2,None),img_shape=(w1_shape[2],w1_shape[3]),tile_shape=(w1_shape[0]/8,8),tile_spacing=(1,1)))
    image.save('filters.png')





