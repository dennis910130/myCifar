__author__ = 'chensi'
import numpy
import cPickle
import optparse


def main():
    parser = optparse.OptionParser()
    parser.add_option("-f",dest="filename")

    (options, args) = parser.parse_args()
    f_out = open('output.txt','wb')
    f = open(options.filename,'rb')
    temp = cPickle.load(f)
    f.close()

    titles = ['img_input','conv_pool1.output','conv_pool2.output','conv_pool3.output','fc_64.output','fc_10.p_y_given_x']
    for i in range(len(titles)):
        f_out.write(titles[i]+':\n')
        temp_i=numpy.array(temp[i])
        if temp_i.size>200:
            if len(temp_i.shape) == 4:
                f_out.write(str(temp_i[:,:,:,0]))
            elif len(temp_i.shape) == 3:
                f_out.write(str(temp_i[:,:,0]))
            elif len(temp_i.shape) == 2:
                f_out.write(str(temp_i[:,0]))
            else:
                f_out.write(str(temp_i))
            f_out.write('\n')
        else:
            f_out.write(str(temp_i))
            f_out.write('\n')

    f_out.close()






if __name__ == '__main__':
    main()
