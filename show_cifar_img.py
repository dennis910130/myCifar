__author__ = 'chensi'
import cPickle
import numpy
from utils import tile_raster_images
import PIL.Image

data_path = ''
def show_cifar_img():
    f = open(data_path + 'data_batch_1','rb')
    img = cPickle.load(f)
    data = img['data']
    labels = numpy.array(img['labels'])
    data = data.reshape(10000,3,1024)

    for i in range(10):
        index = labels == i
        temp_data = data[index,:,:]
        temp_data = temp_data[0:100,:,:]
        temp_data = temp_data + 0.0
        print temp_data.dtype
        x0 = temp_data[:,0,:]
        x1 = temp_data[:,1,:]
        x2 = temp_data[:,2,:]

        image = PIL.Image.fromarray(tile_raster_images(X=(x0,x1,x2,None),
                                                       img_shape=(32,32),
                                                       tile_shape=(10,10),
                                                       tile_spacing=(1,1)))
        name = 'class_' + str(i) + '.png'
        image.save(name)

if __name__ == '__main__':
    show_cifar_img()