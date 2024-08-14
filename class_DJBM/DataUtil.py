import os
from time import time
from DataHelper import DataHelper

class DataUtil():
    def __init__(self, conf):
        self.conf = conf
        #print('DataUtil, Line12, test- conf data_dir:%s' % self.conf.data_dir)

    def initializeRankingHandle(self):
        #t0 = time()
        self.createTrainHandle()
        self.createEvaluateHandle()
        #t1 = time()
        #print('Prepare data cost:%.4fs' % (t1 - t0))
    
    def createTrainHandle(self):
        data_dir = self.conf.data_dir

        train_rating_filename = "%s/%s.train.rating" % (data_dir, self.conf.data_name)
        test_rating_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)
        train_link_filename = "%s/%s.train.link" % (data_dir, self.conf.data_name)
        test_link_filename = "%s/%s.test.link" % (data_dir, self.conf.data_name)

        self.train_rating = DataHelper(self.conf, train_rating_filename)
        self.test_rating = DataHelper(self.conf, test_rating_filename)
        self.train_social = DataHelper(self.conf, train_link_filename)
        self.test_social = DataHelper(self.conf, test_link_filename)

    def createEvaluateHandle(self):
        data_dir = self.conf.data_dir

        test_rating_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)
        test_link_filename = "%s/%s.test.link" % (data_dir, self.conf.data_name)

        self.test_rating_eva = DataHelper(self.conf, test_rating_filename)
        self.test_social_eva = DataHelper(self.conf, test_link_filename)
