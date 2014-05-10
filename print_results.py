__author__ = 'chensi'
import numpy
import cPickle
import optparse


def main():
    parser = optparse.OptionParser()
    parser.add_option("-f",dest="filename")
    parser.add_option("-s",dest="Saliency",default=False)
    (options, args) = parser.parse_args()
    f_out = open('f_out.txt','wb')
    f = open(options.filename,'rb')
    temp = cPickle.load(f)
    f.close()
    if options.Saliency:
        titles = ['layer1.W','layer1.b','layer2.W','layer2.b','layer3.W','layer3.b','fc1.W','fc1.b','fc2.W','fc2.b',
                  'layer1.T','layer2.T','layer3.T','layer1.Gamma','layer2.Gamma','layer3.Gamma','wcs','epsGs','epsTs','epsWs','epsBs','momGs','momTs','momWs','momBs']
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
    else:
        titles = ['layer1.W','layer1.b','layer2.W','layer2.b','layer3.W','layer3.b','fc1.W','fc1.b','fc2.W','fc2.b',
                  'momWs','momBs','epsWs','epsBs','wcs']

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