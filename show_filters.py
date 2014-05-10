__author__ = 'chensi'
from myUtils import filter_visualize_color
try:

    filter_visualize_color('current_best_params.pkl')
except:
    print 'failed'
