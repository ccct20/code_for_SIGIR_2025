#coding=utf-8
from __future__ import division
import os, sys, shutil


from time import time
import numpy as np
import tensorflow as tf
from ipdb import set_trace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #ignore the warnings 

from Logging import Logging

def start(conf, data, model, evaluate):
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # define log name 
    log_path = os.path.join(os.getcwd(),'log/gowalla_balance_参数修改后log/','%s_%s_%sfactor_%sdepth_%sneg_%s_att_att.log'\
                             % (conf.data_name, conf.model_name, conf.k, conf.layer_depth, conf.num_social_negatives, conf.k_2order_negtive))

    data.initializeRankingHandle()


    print('System start to load data...')
    t0 = time()
    d_train_rating, d_train_social = data.train_rating, data.train_social
    d_train_rating.initializeRatingTrain("r_train")
    d_train_social.initializeSocialTrain("s_train")

    train_social_hash_data = d_train_social.social_hash_data

    d_test_rating, d_test_social = data.test_rating, data.test_social
    d_test_rating.initializeRatingTestLoss('r_test')
    d_test_social.initializeSocialTestLoss('s_test')
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))


    d_test_eva_rating, d_test_eva_social = data.test_rating_eva, data.test_social_eva
    d_test_eva_rating.initalizeRatingEva()
    d_test_eva_social.initalizeSocialEva(train_social_hash_data)


    print("Start prepareModelSupplement on data_dict tor training")
    t1=time()
    data_dict = d_train_rating.prepareModelSupplement(model)
    t2=time()
    print("**** prepare rating training data in train.py ,cost: %f ****"%(t2-t1))


    print("Start prepareModelSupplement on data_dict tor social training")
    t1=time()
    s_data_dict = d_train_social.prepareModelSocialSupplement(model)
    d_train_social.generateSocialTrainNegative()
    t2=time()
    print("**** prepare social training data in train.py cost: %f ****"%(t2-t1))


    print("Start updating data_dict")
    t0=time()
    data_dict.update(s_data_dict) # merge social data_dict and rating data_dict
    t1=time()
    print("**** data_dict updating cost: %f ****"%(t1-t0))


    print("Start input supply data_dict")
    t0=time()
    model.inputSupply(data_dict)
    t1=time()
    print("**** inputsupply cost: %f ****"%(t1-t0))


    print("Start construct model graph")
    t1=time()
    model.startConstructGraph()
    t2=time()
    print("**** startConstructGraph in train.py cost: %f ****"%(t2-t1))


    # standard tensorflow running environment initialize
    tf_conf = tf.ConfigProto()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_conf)
    sess.run(model.init)

    # the model save path
    save_path = os.path.join(os.getcwd(),'model/2/','%s_%s_%sfactor_%sdepth_%sneg_%s_model.ckpt' % (conf.data_name, conf.model_name, conf.k, conf.layer_depth, conf.num_social_negatives, conf.k_2order_negtive))
    saver = tf.train.Saver()    
    
    # load graph structure
    # saver = tf.train.import_meta_graph(save_path +'.meta')
    
    # save model
    model_path = saver.save(sess, save_path)
    print("Model saved in path: %s" % model_path)
    
    # load model
    if conf.pretrain_flag == 1:
        saver.restore(sess, save_path)

    
    # set debug_flag=0, doesn't print any results
    log = Logging(log_path)
    log.record('Following will output the evaluation of the model:')

    
    # Start Training !!!
    for epoch in range(1, conf.epochs+1):
        # optimize model with training data and compute train loss
        tmp_train_loss = []
        tmp_r_loss = []
        tmp_s_loss = []
        t0 = time()
        
        #print ("terminal_flag:%f"%d_train_rating.terminal_flag)
        #print ("index:%d"%d_train_rating.index)

        while d_train_rating.terminal_flag:

            d_train_rating.getTrainRatingBatch()
            d_train_rating.linkedMapRatingTrain()

            d_train_social.getTrainSocialBatch()
            d_train_social.linkedMapSocialTrain()
            
            train_feed_dict = {}
            
            for (key, value) in model.map_dict['r_train'].items():
                train_feed_dict[key] = d_train_rating.data_dict[value]
            for (key, value) in model.map_dict['s_train'].items():
                train_feed_dict[key] = d_train_social.data_dict[value]
     
            flag = 1

            [sub_train_loss,_,r_loss,s_loss,low_att_user_user,low_att_user_item,low_att_user_item_show,low_att_user_item_sparse_matrix] = sess.run([model.map_dict['out']['total_train_loss'], model.opt, \
                                                        model.map_dict['out']['r_loss'],model.map_dict['out']['s_loss'],model.map_dict['out']['low_att_user_user']\
                                                        ,model.map_dict['out']['low_att_user_item'],model.map_dict['out']['low_att_user_item_show'],\
                                                        model.map_dict['out']['low_att_user_item_sparse_matrix']], feed_dict=train_feed_dict)
            
            
            
            #print r_loss
            #print s_loss
            #print d_train_rating.index
            tmp_train_loss.append(sub_train_loss)
            tmp_r_loss.append(r_loss)
            tmp_s_loss.append(s_loss)
            '''
            print("low_att_user_user_mean:", low_att_user_user[0])
            print("low_att_user_user_var:", low_att_user_user[1])
            '''

            print("low_att_user_item_mean:", low_att_user_item[0])
            print("low_att_user_item_var:", low_att_user_item[1])
        #set_trace()
        print(np.mean(low_att_user_item[0]),np.mean(low_att_user_item[1]))
        #set_trace()
        train_loss = np.mean(tmp_train_loss)
        t1 = time()
        print("**** Current epoch training time cost:%f ****"%(t1-t0))

        print("rating loss:%f"%(np.mean(tmp_r_loss)))
        print("social loss:%f"%(np.mean(tmp_s_loss)))
        #print train_loss
        
        # compute val loss and test loss
        
        '''
        d_val.getVTRankingOneBatch()
        d_val.linkedMap()
        val_feed_dict = {}
        for (key, value) in model.map_dict['val'].items():
            val_feed_dict[key] = d_val.data_dict[value]
        val_loss = sess.run(model.map_dict['out']['val'], feed_dict=val_feed_dict)
        '''

        t_test1 = time()
        
        d_test_rating.getOneBatchForRatingTestLoss()
        d_test_rating.linkedMapRatingTestLoss()

        #set_trace()

        d_test_social.getOneBatchForSocialTestLoss()
        d_test_social.linkedMapSocialTestLoss()

        r_test_feed_dict = {}

        s_test_feed_dict = {}

        for (key, value) in model.map_dict['r_test'].items():
            r_test_feed_dict[key] = d_test_rating.data_dict[value]

        for (key, value) in model.map_dict['s_test'].items():
            s_test_feed_dict[key] = d_test_social.data_dict[value]

        #rating test loss
        test_loss,rmse = sess.run([model.map_dict['out']['r_test'],model.map_dict['out']['rmse']], feed_dict=r_test_feed_dict)
        
        #social test loss
        s_test_loss = sess.run(model.map_dict['out']['s_test'], feed_dict=s_test_feed_dict)

        #set_trace()
        '''
        for (key, value) in model.map_dict['r_test'].items():
            r_test_feed_dict[key] = d_test_rating.data_dict[value]
        test_loss,rmse = sess.run([model.map_dict['out']['r_test'],model.map_dict['out']['rmse']], feed_dict=r_test_feed_dict)
        '''
        log.record("RMSE:%f"%(rmse))
        t2 = time()
        

        # start evaluate model performance, hr and ndcg
        def getPositiveRatingPredictions():
            # RATING 
            d_test_eva_rating.getEvaRatingPositiveBatchOneTime()
            d_test_eva_rating.linkedRatingEvaMap()

            #set_trace()
            r_eva_feed_dict = {}

            for (key, value) in model.map_dict['r_eva'].items():
                r_eva_feed_dict[key] = d_test_eva_rating.data_dict[value]

            r_positive_predictions = sess.run(
                model.map_dict['out']['r_eva'],
                feed_dict=r_eva_feed_dict
            )

            return r_positive_predictions

        def getPositiveSocialPredictions():
            # SOCIAL
            d_test_eva_social.getEvaSocialPositiveBatchOneTime()
            d_test_eva_social.linkedSocialEvaMap()
            s_eva_feed_dict = {}

            for (key, value) in model.map_dict['s_eva'].items():
                s_eva_feed_dict[key] = d_test_eva_social.data_dict[value]

            s_positive_predictions = sess.run(
                model.map_dict['out']['s_eva'],
                feed_dict=s_eva_feed_dict
            )
            return s_positive_predictions


        def getNegativeRatingPredictions():
            negative_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, terminal_flag = d_test_eva_rating.getEvaRatingRankingBatch()
                d_test_eva_rating.linkedRatingEvaMap()
                r_eva_feed_dict = {}
                for (key, value) in model.map_dict['r_eva'].items():
                    r_eva_feed_dict[key] = d_test_eva_rating.data_dict[value]
                index = 0
                tmp_negative_predictions = np.reshape(
                    sess.run(
                        model.map_dict['out']['r_eva'],
                        feed_dict=r_eva_feed_dict
                    ),
                    [-1, conf.num_evaluate])
                for u in batch_user_list:
                    negative_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return negative_predictions


        def getNegativeSocialPredictions():
            negative_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, terminal_flag = d_test_eva_social.getEvaSocialRankingBatch()
                d_test_eva_social.linkedSocialEvaMap()

                s_eva_feed_dict = {}
                for (key, value) in model.map_dict['s_eva'].items():
                    s_eva_feed_dict[key] = d_test_eva_social.data_dict[value]
                #set_trace()
                index = 0
                tmp_negative_predictions = np.reshape(
                    sess.run(
                        model.map_dict['out']['s_eva'],
                        feed_dict=s_eva_feed_dict
                    ),
                    [-1, conf.num_social_evaluate])

                #set_trace()
                for u in batch_user_list:
                    negative_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return negative_predictions


        tt2 = time()

        r_index_dict = d_test_eva_rating.eva_rating_index_dict
        s_index_dict = d_test_eva_social.eva_social_index_dict


        r_positive_predictions = getPositiveRatingPredictions()
        r_negative_predictions = getNegativeRatingPredictions()

        
        s_positive_predictions = getPositiveSocialPredictions()
        s_negative_predictions = getNegativeSocialPredictions()
        
        #print(np.mean(positive_predictions))


        #set_trace()
        d_test_eva_rating.index = 0 # !!!important, prepare for new batch
        d_test_eva_social.index = 0

        
        #set_trace()
        s_hr_10, s_ndcg_10, s_pre_10, s_rec_10, s_f1_10 = evaluate.evaluateRankingPerformance(\
            s_index_dict, s_positive_predictions, s_negative_predictions, conf.topk, conf.num_procs)
        s_hr_5, s_ndcg_5, s_pre_5, s_rec_5, s_f1_5 = evaluate.evaluateRankingPerformance(\
            s_index_dict, s_positive_predictions, s_negative_predictions, conf.top5, conf.num_procs)
        s_hr_15, s_ndcg_15, s_pre_15, s_rec_15, s_f1_15 = evaluate.evaluateRankingPerformance(\
            s_index_dict, s_positive_predictions, s_negative_predictions, conf.top15, conf.num_procs)

        hr_10, ndcg_10, pre_10, rec_10, f1_10 = evaluate.evaluateRankingPerformance(\
            r_index_dict, r_positive_predictions, r_negative_predictions, conf.topk, conf.num_procs)
        hr_5, ndcg_5, pre_5, rec_5, f1_5 = evaluate.evaluateRankingPerformance(\
            r_index_dict, r_positive_predictions, r_negative_predictions, conf.top5, conf.num_procs)
        hr_15, ndcg_15, pre_15, rec_15, f1_15 = evaluate.evaluateRankingPerformance(\
            r_index_dict, r_positive_predictions, r_negative_predictions, conf.top15, conf.num_procs)

        tt3 = time()

                
        # print log to console and log_file
        
        log.record('Epoch:%d, compute loss cost:%.4fs, train loss:%.4f, rating test loss:%.4f, social test loss:%.4f' % \
            (epoch, (t2-t0), train_loss, test_loss, s_test_loss))

        '''
        log.record('Evaluate cost:%.4fs \n \
                    "Rating: \t\t Social: \n \
                    "Top5: hr:%.4f, ndcg:%.4f \t Top5: hr:%.4f, ndcg:%.4f  \n \
                    Top10: hr:%.4f, ndcg:%.4f \t Top10: hr:%.4f, ndcg:%.4f \n \
                    Top15: hr:%.4f, ndcg:%.4f \t Top15: hr:%.4f, ndcg:%.4f ' % ((tt3-tt2), hr_5, ndcg_5, s_hr_5, \
                                                                                               s_ndcg_5, hr_10, ndcg_10, s_hr_10, s_ndcg_10, \
                                                                                               hr_15, ndcg_15,s_hr_15, s_ndcg_15))
        '''
        
        log.record('Evaluate cost:%.4fs \n \
                    "Rating: \t\t Social: \n \
                    "Top5: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \t Top5: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f  \n \
                    Top10: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \t Top10: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \n \
                    Top15: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \t Top15: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f ' % ((tt3-tt2), hr_5, ndcg_5, pre_5, rec_5, f1_5, s_hr_5, s_ndcg_5, s_pre_5, s_rec_5, s_f1_5, \
                                                                                                                                                         hr_10, ndcg_10, pre_10, rec_10, f1_10, s_hr_10, s_ndcg_10, s_pre_10, s_rec_10, s_f1_10,\
                                                                                                                                                         hr_15, ndcg_15, pre_15, rec_15, f1_15, s_hr_15, s_ndcg_15, s_pre_15, s_rec_15, s_f1_15))
        t_test2 = time()       
        print("**** current epoch test time cost:%f ****"%(t_test2-t_test1))   
        log.record('------------------------------------------------------------------')


        d_train_rating.generateRatingTrainNegative()
        d_train_social.generateSocialTrainNegative()


















