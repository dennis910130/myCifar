__author__ = 'chensi'
import numpy
import cPickle
import time
def params_estimation(mu=0.001,eita=1.0,beta=1.0):
    data_path = '/home/chensi/mylocal/sichen/data/class_response/'
    time1 = time.time()
    alphas = numpy.zeros((10,32))
    for class_i in range(10):
    print 'class:' + str(class_i) + '...'
        pkl_file_name = data_path + 'class_' + str(class_i) + '.pkl'
        file = open(pkl_file_name,'rb')
        temp = cPickle.load(file)
        file.close()
        for filter_i in range(32):
            print 'filter:' + str(filter_i) + '...'
            filter_response = temp[:,filter_i,:].reshape((-1,))
            n = len(filter_response)
            kappa = (n+eita)/beta
            filter_response = numpy.abs(filter_response)
            filter_response = numpy.power(filter_response,beta)
            alphas[class_i,filter_i] = 1.0/kappa*(numpy.sum(filter_response)+mu)
        print 'done...'
    
    alpha_path = data_path + 'alphas.pkl'
    print 'saving to file %s' % alpha_path
    file = open(alpha_path,'wb')
    cPickle.dump(alphas,file)
    file.close()
    duration = time.time() - time1
    print 'Run for %f mins' % (duration/60.0)

if __name__ == '__main__':
    params_estimation()
