__author__ = 'chensi'
import numpy as np
import sys
caffe_root = ''
sys.path.insert(0,caffe_root + 'python')
import caffe
import glob
imageset_path = ''
import cPickle
def main():
    net = caffe.Classifier(caffe_root+'examples/imagenet/imagenet_deploy.prototxt',
                           caffe_root+'examples/imagenet/caffe_reference_imagenet_model')
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data',caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    net.set_channel_swap('data',(2,1,0))
    net.set_input_scale('data',255)

    file_names = glob.glob(imageset_path+'*.jpg')
    features = np.zeros((len(file_names),10,256,13,13))
    for i in range(len(file_names)):
        net.predict([caffe.io.load_image(file_names[i])])
        feature = net.blobs['conv5'].data
        print feature.shape
        features[i,:] = feature
    out_file = open(imageset_path+'CNN_features.pkl','wb')
    cPickle.dump(features,out_file,protocol=cPickle.HIGHEST_PROTOCOL)
    out_file.close()

if __name__ == '__main__':
    main()





