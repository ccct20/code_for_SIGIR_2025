#coding=utf-8
from __future__ import division
from collections import defaultdict
import numpy as np
from time import time
import random
import tensorflow as tf
from ipdb import set_trace
import os


def Normalization_gowalla(inX):
        return 1.0 / (1 + np.exp(-inX))

class DataHelper():
    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0

    
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_ITEMS_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrix()
            #self.arrangePositiveData()
            data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_Tdict_list
            data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_Tdict_list


        if 'ITEM_CUSTOMER_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrixForItemUser()      
            data_dict['ITEM_CUSTOMER_INDICES_INPUT'] = self.item_customer_indices_Tdict_list
            data_dict['ITEM_CUSTOMER_VALUES_INPUT'] = self.item_customer_values_Tdict_list 
            #data_dict['ITEM_USER_NUM_DICT_INPUT'] = self.item_user_num_dict  #for sparsity

        return data_dict

    def prepareModelSocialSupplement(self, model):
        data_dict = {}
        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            #self.ReadSocialData()
            self.generateSocialNeighborsSparseMatrix()
            self.arrangeBalanceSocialData()
            self.generateBalanceSocialNeighborsSparseMatrix()
            data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_Tdict_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_Tdict_list

            data_dict['FIRST_TRUST_SECOND_TRUST_NEIGHBORS_INDICES_INPUT'] = self.first_trust_second_trust_neighbors_Tdict_indices_Tdict_list
            data_dict['FIRST_TRUST_SECOND_TRUST_NEIGHBORS_VALUES_INPUT'] = self.first_trust_second_trust_neighbors_Tdict_values_Tdict_list

            data_dict['FIRST_DIS_SECOND_DIS_NEIGHBORS_INDICES_INPUT'] = self.first_dis_second_dis_neighbors_Tdict_indices_Tdict_list
            data_dict['FIRST_DIS_SECOND_DIS_NEIGHBORS_VALUES_INPUT'] = self.first_dis_second_dis_neighbors_Tdict_values_Tdict_list
            #data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'] = self.social_neighbor_num_list
            #data_dict['SOCIAL_NEIGHBORS_NUM_DICT_INPUT'] = self.social_neighbors_num_dict
            #data_dict['USER_USER_SPARSITY_DICT']= self.user_user_sparsity_dict


        return data_dict

# ==================== initalize operation ==================== #

    # RATING
    def initalizeRatingEva(self):
        self.ReadRatingData()
        self.getEvaRatingPositiveBatchOneTime()
        self.generateEvaRatingNegative()


    def initializeRatingTestLoss(self,flag):
        self.ReadRatingData()
        self.arrangePositiveRatingData(flag)
        self.arrangePositiveRatingDataForItemUser(flag)
        self.generateRatingTestNegative()
        #self.generateConsumedItemsSparseMatrix()
        #self.generateConsumedItemsSparseMatrixForItemUser()

    def initializeRatingTrain(self,flag):
        #self.ReadRatingData()
        self.ReadRatingData()
        self.arrangePositiveRatingData(flag)
        self.arrangePositiveRatingDataForItemUser(flag)
        self.generateRatingTrainNegative() # neg_num=0


        #self.generateConsumedItemsSparseMatrix()
        #self.generateConsumedItemsSparseMatrixForItemUser()
        #self.GetRatingMissingPeriod()
        #self.arrangePositiveRatingDataForUser()

    def linkedMapRatingTrain(self):
        self.data_dict['USER_LIST'] = self.b_rating_user_list
        self.data_dict['ITEM_LIST'] = self.b_rating_item_list
        self.data_dict['LABEL_LIST'] = self.b_rating_labels_list
        self.data_dict['STAMP_LIST'] = self.b_rating_stamp_list
    def linkedMapRatingTestLoss(self):
        self.data_dict['TEST_USER_LIST'] = self.test_rating_user_list
        self.data_dict['TEST_ITEM_LIST'] = self.test_rating_item_list
        self.data_dict['TEST_LABEL_LIST'] = self.test_rating_labels_list
    def linkedRatingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list
        #self.data_dict['EVA_LABEL_LIST'] = self.test_rating_labels_list


    # SOCIAL
    def initializeSocialTrain(self, flag):
        #self.ReadRatingData()
        self.ReadSocialData()
        self.arrangePositiveSocialData(flag)
        # self.generateSocialTrainNegative()

    def initalizeSocialEva(self,train_hash_data):
        self.ReadSocialData()
        self.getEvaSocialPositiveBatchOneTime()
        self.generateEvaSocialNegative(train_hash_data)

    def initializeSocialTestLoss(self, flag):
        self.ReadSocialData()
        self.arrangePositiveSocialData(flag)
        self.generateSocialTestNegative()


    def linkedMapSocialTrain(self):
        self.data_dict['USER1_LIST'] = self.b_social_user1_list
        self.data_dict['USER2_LIST'] = self.b_social_user2_list
        self.data_dict['S_LABEL_LIST'] = self.b_social_labels_list
        self.data_dict['S_STAMP_LIST'] = self.b_social_stamp_list
    def linkedSocialEvaMap(self):
        self.data_dict['EVA_USER1_LIST'] = self.eva_user1_list
        self.data_dict['EVA_USER2_LIST'] = self.eva_user2_list
        #self.data_dict['EVAL_s_LABEL_LIST'] = self.eva_s_labels_input
    def linkedMapSocialTestLoss(self):
        self.data_dict['TEST_USER1_LIST'] = self.test_social_user1_list
        self.data_dict['TEST_USER2_LIST'] = self.test_social_user2_list
        self.data_dict['TEST_S_LABEL_LIST'] = self.test_social_labels_list




# ==================== First Part Data reading relative ==================== #
# Data loading as list, dict, set, reporting statistic, etc.

    def ReadSocialData(self):
        print("**** Start Loading Link Data ****")
        tsl1 = time()
        f = open(self.filename)
        total_link_user1_list = set()
        total_link_user2_list = set()
        user_link_stamp_dict = {}
        social_hash_Tdict_interaction = {}
        

        hash_data = defaultdict(int) #hash_data = 0
        hash_data_T = defaultdict(int)

        stamp_hash_data = defaultdict(int)
        
        for _, line in enumerate(f):

            arr = line.split("\t")
            u1 = int(arr[0])
            u2 = int(arr[1])
            stamp = int(arr[2])

            hash_data[u1, u2] = 1
            hash_data_T[u1, u2, stamp] = 1
            
            total_link_user1_list.add(int(arr[0]))
            total_link_user2_list.add(int(arr[1]))

      
            if u1 in user_link_stamp_dict:
                if stamp in user_link_stamp_dict[u1]:
                    flag = 1
                else:
                    user_link_stamp_dict[u1].append((u2,stamp))
            else:
                user_link_stamp_dict[u1]=[]
                user_link_stamp_dict[u1].append((u2,stamp))


            #****** use for negative sampling ******#
            if stamp in social_hash_Tdict_interaction.keys():
                social_hash_Tdict_interaction[stamp].append((u1, u2))
            else:
                social_hash_Tdict_interaction[stamp]=[]
                social_hash_Tdict_interaction[stamp].append((u1, u2))


            


        self.total_social_user1_list = list(total_link_user1_list)
        self.total_social_user2_list = list(total_link_user2_list)
        self.social_hash_data = hash_data
        self.social_hash_data_T = hash_data_T
        self.user_link_stamp_dict = user_link_stamp_dict
        self.social_hash_Tdict_interaction = social_hash_Tdict_interaction
        tsl2 = time()
        print("**** End Loading Link Data(Finish), cost: %f****"%(tsl2-tsl1))




    def ReadRatingData(self):
        #set_trace()
        print("**** Start Loading Rating Data****")
        trl1 = time()
        fliter_r = 0
        f = open(self.filename) ## May should be specific for different subtasks
        total_rating_user_list = set()
        user_rating_stamp_dict = {} 

        hash_data = defaultdict(int) #hash_data = 0
        rating_hash_Tdict_interaction = defaultdict(int)  #for negative sampling
        hash_data_T = defaultdict(int) # for training
        
        for _, line in enumerate(f):
            arr = line.split("\t")
            u = int(arr[0])  
            v = int(arr[1])
            stamp = int(arr[3])
            if self.conf.data_name =='gowalla':
                #r = Normalization_gowalla(float(arr[2]))
                r = float(arr[2])
            else:
                #r = int(float(arr[2]))/self.conf.max_score)
                r = int(float(arr[2])-1.0)/4.0
                #r = int(float(arr[2]))

            
            
            if (r>=fliter_r/self.conf.max_score):

                hash_data[(int(arr[0]), int(arr[1]))] = r
                total_rating_user_list.add(int(arr[0]))

                # ******** for training ******** #
                #set_trace()



                hash_data_T[int(arr[0]), int(arr[1]), r, int(arr[3])] = 1



                #****** use for negative sampling ******#
                if stamp in rating_hash_Tdict_interaction.keys():
                    rating_hash_Tdict_interaction[stamp].append((int(arr[0]), int(arr[1])))
                else:
                    rating_hash_Tdict_interaction[stamp]=[]
                    rating_hash_Tdict_interaction[stamp].append((int(arr[0]), int(arr[1])))


                #****** the periods that users have ******#
                if u in user_rating_stamp_dict:
                    if stamp in user_rating_stamp_dict[u]:
                        flag = 1
                    else:
                        user_rating_stamp_dict[u].append(stamp)
                else:
                    user_rating_stamp_dict[u]=[]
                    user_rating_stamp_dict[u].append(stamp)



            

        self.total_rating_user = list(total_rating_user_list)
        self.rating_hash_data = hash_data
        self.rating_hash_data_T = hash_data_T
       

        self.rating_hash_Tdict_interaction = rating_hash_Tdict_interaction # t:u-v
        self.user_rating_stamp_dict = user_rating_stamp_dict  # u:t


        trl2 = time()
        print("**** End Loading Rating Data(Finish), cost: %f****"%(trl2-trl1))




    def arrangePositiveSocialData(self, flag):  # It has been revised for faster the starting part

        if flag == 's_train':
            print("**** Start Arranging Positive Social Data for training ****")
            t_aps_train1 = time()
            if self.conf.flag_1st_trainset == 0:
                social_positive_data = defaultdict(set)
                social_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                social_positive_data_for_user1_user2_Tdict = {}
                social_positive_data_for_user1_user2_T_Udict = {}
                social_positive_data_for_user1_user2_withoutR_Tdict = {}
                #social_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                social_total_data = set()
                social_total_data_T = set()
                social_hash_data = self.social_hash_data
                social_hash_data_T = self.social_hash_data_T

                for (u1, u2) in social_hash_data:
                    social_total_data.add((u1, u2))
                    social_positive_data[u1].add(u2)

                # ********  the rating total data is used for training ******** #
                for (u1, u2, stamp) in social_hash_data_T:
                    social_total_data_T.add((u1, u2, stamp))
                    social_positive_data_T[u1].add((u2, stamp))

                social_user_list = sorted(list(social_positive_data.keys()))

                self.social_positive_data_for_user1_user2_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_Tdict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_T_Udict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_withoutR_Tdict.npy',allow_pickle=True).item()
                
                
                self.social_user_list = social_user_list

            elif self.conf.flag_1st_trainset == 1:

                social_positive_data = defaultdict(set)
                social_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                social_positive_data_for_user1_user2_Tdict = {}
                social_positive_data_for_user1_user2_T_Udict = {}
                social_positive_data_for_user1_user2_withoutR_Tdict = {}
                #social_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                social_total_data = set()
                social_total_data_T = set()
                social_hash_data = self.social_hash_data
                social_hash_data_T = self.social_hash_data_T

                for (u1, u2) in social_hash_data:
                    social_total_data.add((u1, u2))
                    social_positive_data[u1].add(u2)

                # ********  the rating total data is used for training ******** #
                for (u1, u2, stamp) in social_hash_data_T:
                    social_total_data_T.add((u1, u2, stamp))
                    social_positive_data_T[u1].add((u2, stamp))

                social_user_list = sorted(list(social_positive_data.keys()))

                relation = 1.0

                #  for GCN propagation computing:  Key:T    --> key U  ---> (i,r)

                
                
                for (u1, u2, t) in social_hash_data_T:

                    if t in social_positive_data_for_user1_user2_Tdict.keys():
                        if u1 in social_positive_data_for_user1_user2_Tdict[t].keys():
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                        else:
                            social_positive_data_for_user1_user2_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                    else:
                        social_positive_data_for_user1_user2_Tdict[t]={}
                        if u1 in social_positive_data_for_user1_user2_Tdict[t].keys():
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                        else:
                            social_positive_data_for_user1_user2_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                


                #  for GCN propagation computing:  Key:T    --> key U  ---> (i list)

                
                for (u1, u2, t) in social_hash_data_T:  
                    

                    if t in social_positive_data_for_user1_user2_withoutR_Tdict.keys():
                        if u1 in social_positive_data_for_user1_user2_withoutR_Tdict[t].keys():
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                        else:
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                    else:
                        social_positive_data_for_user1_user2_withoutR_Tdict[t]={}
                        if u1 in social_positive_data_for_user1_user2_withoutR_Tdict[t].keys():
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                        else:
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                            

                #  for training positive data:  Key:u    --> (i,r,t)
                
                
                for (u1, u2, t) in social_hash_data_T:
                    if u1 in social_positive_data_for_user1_user2_T_Udict.keys():
                        social_positive_data_for_user1_user2_T_Udict[u1].append((u2, relation, t))

                    else:
                        social_positive_data_for_user1_user2_T_Udict[u1]=[]
                        social_positive_data_for_user1_user2_T_Udict[u1].append((u2, relation, t))
                
                


                self.social_positive_data = social_positive_data
                self.social_positive_data_T = social_positive_data_T

                self.social_total_data = len(social_total_data)
                self.social_total_data_T = len(social_total_data_T)

                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_Tdict.npy',social_positive_data_for_user1_user2_Tdict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_T_Udict.npy',social_positive_data_for_user1_user2_T_Udict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_withoutR_Tdict.npy',social_positive_data_for_user1_user2_withoutR_Tdict)
            
                self.social_positive_data_for_user1_user2_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_Tdict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_T_Udict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_social_positive_data_for_user1_user2_withoutR_Tdict.npy',allow_pickle=True).item()

                self.conf.flag_1st_trainset = 0
            self.social_user_list = social_user_list
            t_aps_train2 = time()
            
            print("**** End Arranging Positive Social Data for training, cost:%f ****"%(t_aps_train2-t_aps_train1))

        if flag == 's_test':
   
            print("**** Start Arranging Positive Social Data for training ****")
            t_aps_test1 = time()

            if self.conf.flag_1st_testset == 0:
                social_positive_data = defaultdict(set)
                social_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                social_positive_data_for_user1_user2_Tdict = {}
                social_positive_data_for_user1_user2_T_Udict = {}
                social_positive_data_for_user1_user2_withoutR_Tdict = {}
                #social_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                social_total_data = set()
                social_total_data_T = set()
                social_hash_data = self.social_hash_data
                social_hash_data_T = self.social_hash_data_T

                for (u1, u2) in social_hash_data:
                    social_total_data.add((u1, u2))
                    social_positive_data[u1].add(u2)

                # ********  the rating total data is used for training ******** #
                for (u1, u2, stamp) in social_hash_data_T:
                    social_total_data_T.add((u1, u2, stamp))
                    social_positive_data_T[u1].add((u2, stamp))

                social_user_list = sorted(list(social_positive_data.keys()))

                self.social_positive_data_for_user1_user2_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_Tdict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_T_Udict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_withoutR_Tdict.npy',allow_pickle=True).item()   

                self.social_user_list = social_user_list 

            elif self.conf.flag_1st_testset == 1:

                social_positive_data = defaultdict(set)
                social_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                social_positive_data_for_user1_user2_Tdict = {}
                social_positive_data_for_user1_user2_T_Udict = {}
                social_positive_data_for_user1_user2_withoutR_Tdict = {}
                #social_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                social_total_data = set()
                social_total_data_T = set()
                social_hash_data = self.social_hash_data
                social_hash_data_T = self.social_hash_data_T

                for (u1, u2) in social_hash_data:
                    social_total_data.add((u1, u2))
                    social_positive_data[u1].add(u2)

                # ********  the rating total data is used for training ******** #
                for (u1, u2, stamp) in social_hash_data_T:
                    social_total_data_T.add((u1, u2, stamp))
                    social_positive_data_T[u1].add((u2, stamp))

                social_user_list = sorted(list(social_positive_data.keys()))

                relation = 1.0


                #  for GCN propagation computing:  Key:T    --> key U  ---> (i,r)

                
                
                for (u1, u2, t) in social_hash_data_T:

                    if t in social_positive_data_for_user1_user2_Tdict.keys():
                        if u1 in social_positive_data_for_user1_user2_Tdict[t].keys():
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                        else:
                            social_positive_data_for_user1_user2_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                    else:
                        social_positive_data_for_user1_user2_Tdict[t]={}
                        if u1 in social_positive_data_for_user1_user2_Tdict[t].keys():
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                        else:
                            social_positive_data_for_user1_user2_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_Tdict[t][u1].append((u2,relation))
                


                #  for GCN propagation computing:  Key:T    --> key U  ---> (i list)

                
                for (u1, u2, t) in social_hash_data_T:  

                    if t in social_positive_data_for_user1_user2_withoutR_Tdict.keys():
                        if u1 in social_positive_data_for_user1_user2_withoutR_Tdict[t].keys():
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                        else:
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                    else:
                        social_positive_data_for_user1_user2_withoutR_Tdict[t]={}
                        if u1 in social_positive_data_for_user1_user2_withoutR_Tdict[t].keys():
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                        else:
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]=[]
                            social_positive_data_for_user1_user2_withoutR_Tdict[t][u1].append(u2)
                

                #  for training positive data:  Key:u    --> (i,r,t)
                
                
                for (u1, u2, t) in social_hash_data_T:
                    if u1 in social_positive_data_for_user1_user2_T_Udict.keys():
                        social_positive_data_for_user1_user2_T_Udict[u1].append((u2, relation, t))

                    else:
                        social_positive_data_for_user1_user2_T_Udict[u1]=[]
                        social_positive_data_for_user1_user2_T_Udict[u1].append((u2, relation, t))
                
                


                self.social_positive_data = social_positive_data
                self.social_positive_data_T = social_positive_data_T

                self.social_total_data = len(social_total_data)
                self.social_total_data_T = len(social_total_data_T)

                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_Tdict.npy',social_positive_data_for_user1_user2_Tdict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_T_Udict.npy',social_positive_data_for_user1_user2_T_Udict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_withoutR_Tdict.npy',social_positive_data_for_user1_user2_withoutR_Tdict)
                
                self.social_positive_data_for_user1_user2_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_Tdict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_T_Udict.npy',allow_pickle=True).item()
                self.social_positive_data_for_user1_user2_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_social_positive_data_for_user1_user2_withoutR_Tdict.npy',allow_pickle=True).item()


                self.social_user_list = social_user_list   
                #self.conf.flag_1st_testset = 0        
            t_aps_test2 = time()
            #t_apr2 = time()
            
            print("**** End Arranging Positive Social Data for test, cost:%f ****"%(t_aps_test2-t_aps_test1))



    def arrangePositiveRatingData(self, flag):  # rating has already revised for faster the starting part

        #print("**** Start Arranging Positive Rating Data ****")
        #t_apr1 = time()

        if flag == 'r_train':  
            t_apr_train1 = time()   
            print("**** Start Arranging Positive Rating Data for training ****")
            if self.conf.flag_1st_trainset == 0:
                rating_positive_data = defaultdict(set)
                rating_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                rating_positive_data_for_user_item_Tdict = {}
                rating_positive_data_for_user_item_T_Udict = {}
                rating_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                rating_total_data = set()
                rating_total_data_T = set()
                rating_hash_data = self.rating_hash_data
                rating_hash_data_T = self.rating_hash_data_T

                for (u, i) in rating_hash_data:
                    rating_total_data.add((u, i))
                    rating_positive_data[u].add(i)

                # ********  the rating total data is used for training ******** #
                for (u, i, r, t) in rating_hash_data_T:
                    rating_total_data_T.add((u, i, r, t))
                    rating_positive_data_T[u].add((i, r, t))

                rating_user_list = sorted(list(rating_positive_data.keys()))

                self.rating_positive_data_for_user_item_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_Tdict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_T_Udict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_withoutR_Tdict.npy',allow_pickle=True).item()   
                
                self.rating_user_list = rating_user_list

            elif self.conf.flag_1st_trainset == 1:
                rating_positive_data = defaultdict(set)
                rating_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                rating_positive_data_for_user_item_Tdict = {}
                rating_positive_data_for_user_item_T_Udict = {}
                rating_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                rating_total_data = set()
                rating_total_data_T = set()
                rating_hash_data = self.rating_hash_data
                rating_hash_data_T = self.rating_hash_data_T

                for (u, i) in rating_hash_data:
                    rating_total_data.add((u, i))
                    rating_positive_data[u].add(i)

                # ********  the rating total data is used for training ******** #
                for (u, i, r, t) in rating_hash_data_T:
                    rating_total_data_T.add((u, i, r, t))
                    rating_positive_data_T[u].add((i, r, t))

                rating_user_list = sorted(list(rating_positive_data.keys()))

                #  for GCN propagation computing:  Key:T    --> key U  ---> (i,r)

                for (u, i, r, t) in rating_hash_data_T:

                    if t in rating_positive_data_for_user_item_Tdict.keys():
                        if u in rating_positive_data_for_user_item_Tdict[t].keys():
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))
                        else:
                            rating_positive_data_for_user_item_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))
                    else:
                        rating_positive_data_for_user_item_Tdict[t]={}
                        if u in rating_positive_data_for_user_item_Tdict[t].keys():
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))
                        else:
                            rating_positive_data_for_user_item_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))

                #  for GCN propagation computing:  Key:T    --> key U  ---> (i list)
                for (u, i, r, t) in rating_hash_data_T:  

                    if t in rating_positive_data_for_user_item_withoutR_Tdict.keys():
                        if u in rating_positive_data_for_user_item_withoutR_Tdict[t].keys():
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)
                        else:
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)
                    else:
                        rating_positive_data_for_user_item_withoutR_Tdict[t]={}
                        if u in rating_positive_data_for_user_item_withoutR_Tdict[t].keys():
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)
                        else:
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)

                #  for training positive data:  Key:u    --> (i,r,t)
                for (u, i, r, t) in rating_hash_data_T:
                    if u in rating_positive_data_for_user_item_T_Udict.keys():
                        rating_positive_data_for_user_item_T_Udict[u].append((i, r, t))

                    else:
                        rating_positive_data_for_user_item_T_Udict[u]=[]
                        rating_positive_data_for_user_item_T_Udict[u].append((i, r, t))
                
                self.rating_positive_data = rating_positive_data
                self.rating_positive_data_T = rating_positive_data_T

                self.rating_total_data = len(rating_total_data)
                self.rating_total_data_T = len(rating_total_data_T)


                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_Tdict.npy',rating_positive_data_for_user_item_Tdict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_T_Udict.npy',rating_positive_data_for_user_item_T_Udict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_withoutR_Tdict.npy',rating_positive_data_for_user_item_withoutR_Tdict)
                
                self.rating_positive_data_for_user_item_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_Tdict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_T_Udict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'train_rating_positive_data_for_user_item_withoutR_Tdict.npy',allow_pickle=True).item()


                self.rating_user_list = rating_user_list
                #self.conf.flag_1st_trainset = 0
            t_apr_train2 = time()
            print("**** End Arranging Positive Rating Data for training, cost:%f ****"%(t_apr_train2-t_apr_train1))

        if flag == 'r_test':
            t_apr_test1 = time()
            print("**** Start Arranging Positive Rating Data for test ****")
            if self.conf.flag_1st_testset == 0:
                rating_positive_data = defaultdict(set)
                rating_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                rating_positive_data_for_user_item_Tdict = {}
                rating_positive_data_for_user_item_T_Udict = {}
                rating_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                rating_total_data = set()
                rating_total_data_T = set()
                rating_hash_data = self.rating_hash_data
                rating_hash_data_T = self.rating_hash_data_T

                for (u, i) in rating_hash_data:
                    rating_total_data.add((u, i))
                    rating_positive_data[u].add(i)

                # ********  the rating total data is used for training ******** #
                for (u, i, r, t) in rating_hash_data_T:
                    rating_total_data_T.add((u, i, r, t))
                    rating_positive_data_T[u].add((i, r, t))

                rating_user_list = sorted(list(rating_positive_data.keys()))

                self.rating_positive_data_for_user_item_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_Tdict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_T_Udict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_withoutR_Tdict.npy',allow_pickle=True).item()
                self.rating_user_list = rating_user_list

            elif self.conf.flag_1st_testset == 1:

                rating_positive_data = defaultdict(set)
                rating_positive_data_T = defaultdict(set)

                # For GCN propagation computing or training
                rating_positive_data_for_user_item_Tdict = {}
                rating_positive_data_for_user_item_T_Udict = {}
                rating_positive_data_for_user_item_withoutR_Tdict = {}

                #user_item_num_dict = defaultdict(set)             # for different version of GCN 

                rating_total_data = set()
                rating_total_data_T = set()
                rating_hash_data = self.rating_hash_data
                rating_hash_data_T = self.rating_hash_data_T

                for (u, i) in rating_hash_data:
                    rating_total_data.add((u, i))
                    rating_positive_data[u].add(i)

                # ********  the rating total data is used for training ******** #
                for (u, i, r, t) in rating_hash_data_T:
                    rating_total_data_T.add((u, i, r, t))
                    rating_positive_data_T[u].add((i, r, t))

                rating_user_list = sorted(list(rating_positive_data.keys()))

                #  for GCN propagation computing:  Key:T    --> key U  ---> (i,r)

                for (u, i, r, t) in rating_hash_data_T:

                    if t in rating_positive_data_for_user_item_Tdict.keys():
                        if u in rating_positive_data_for_user_item_Tdict[t].keys():
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))
                        else:
                            rating_positive_data_for_user_item_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))
                    else:
                        rating_positive_data_for_user_item_Tdict[t]={}
                        if u in rating_positive_data_for_user_item_Tdict[t].keys():
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))
                        else:
                            rating_positive_data_for_user_item_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_Tdict[t][u].append((i,r))

                #  for GCN propagation computing:  Key:T    --> key U  ---> (i list)
                for (u, i, r, t) in rating_hash_data_T:  

                    if t in rating_positive_data_for_user_item_withoutR_Tdict.keys():
                        if u in rating_positive_data_for_user_item_withoutR_Tdict[t].keys():
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)
                        else:
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)
                    else:
                        rating_positive_data_for_user_item_withoutR_Tdict[t]={}
                        if u in rating_positive_data_for_user_item_withoutR_Tdict[t].keys():
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)
                        else:
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u]=[]
                            rating_positive_data_for_user_item_withoutR_Tdict[t][u].append(i)

                #  for training positive data:  Key:u    --> (i,r,t)
                for (u, i, r, t) in rating_hash_data_T:
                    if u in rating_positive_data_for_user_item_T_Udict.keys():
                        rating_positive_data_for_user_item_T_Udict[u].append((i, r, t))

                    else:
                        rating_positive_data_for_user_item_T_Udict[u]=[]
                        rating_positive_data_for_user_item_T_Udict[u].append((i, r, t))
                
                self.rating_positive_data = rating_positive_data
                self.rating_positive_data_T = rating_positive_data_T

                self.rating_total_data = len(rating_total_data)
                self.rating_total_data_T = len(rating_total_data_T)


                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_Tdict.npy',rating_positive_data_for_user_item_Tdict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_T_Udict.npy',rating_positive_data_for_user_item_T_Udict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_withoutR_Tdict.npy',rating_positive_data_for_user_item_withoutR_Tdict)
                
                self.rating_positive_data_for_user_item_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_Tdict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_T_Udict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_T_Udict.npy',allow_pickle=True).item()
                self.rating_positive_data_for_user_item_withoutR_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'test_rating_positive_data_for_user_item_withoutR_Tdict.npy',allow_pickle=True).item()

                self.rating_user_list = rating_user_list
                #self.conf.flag_1st_testset = 0
            t_apr_test2 = time()
            print("**** End Arranging Positive Rating Data for test, cost:%f ****"%(t_apr_test2-t_apr_test1))
            #t_apr2 = time()
            



    def arrangePositiveRatingDataForItemUser(self,flag):

        print("**** Start Arranging Positive Rating Data for Item User ****")
        t_aprdit1 = time()
       
        
        if flag == 'r_train':
           
            try:
                self.positive_rating_data_for_item_user_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'positive_rating_data_for_item_user_Tdict_train.npy',allow_pickle=True).item()
            except IOError:
                positive_rating_data_for_item_user = {}
                rating_hash_data_T = self.rating_hash_data_T #(2566, 24032, 5, 0)

                # for item-user GCN data preparation
                rating_hash_Tdict_interaction = self.rating_hash_Tdict_interaction  # rating_hash_Tdict_interaction[0]  -> (424, 15178)
             

                rating_total_data_for_item_user = set()
                rating_hash_data_for_item_user = self.rating_hash_data


                # For GCN propagation computing
                for time_stamp in rating_hash_Tdict_interaction.keys():
                    positive_rating_data_for_item_user[time_stamp] = {}
                    for (u, i) in rating_hash_Tdict_interaction[time_stamp]:
                        if i in positive_rating_data_for_item_user[time_stamp].keys():
                            positive_rating_data_for_item_user[time_stamp][i].append(u)
                        else: 
                            positive_rating_data_for_item_user[time_stamp][i] = []
                            positive_rating_data_for_item_user[time_stamp][i].append(u)

                    item_list = sorted(list(positive_rating_data_for_item_user[time_stamp].keys()))

                '''
                for (u, i) in rating_hash_data_for_item_user:
                    rating_total_data_for_item_user.add((i, u))
                    positive_rating_data_for_item_user[i].add(u)
                '''

                self.positive_rating_data_for_item_user_Tdict = positive_rating_data_for_item_user
                #self.rating_total_data_for_item_user = len(rating_total_data_for_item_user)
                #set_trace()
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'positive_rating_data_for_item_user_Tdict_train.npy',positive_rating_data_for_item_user)
                
        
        if flag == 'r_test':
           
            try:
                self.positive_rating_data_for_item_user_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'positive_rating_data_for_item_user_Tdict_test.npy',allow_pickle=True).item()
            except IOError:
                positive_rating_data_for_item_user = {}
                rating_hash_data_T = self.rating_hash_data_T #(2566, 24032, 5, 0)

                # for item-user GCN data preparation
                rating_hash_Tdict_interaction = self.rating_hash_Tdict_interaction  # rating_hash_Tdict_interaction[0]  -> (424, 15178)
             

                rating_total_data_for_item_user = set()
                rating_hash_data_for_item_user = self.rating_hash_data


                # For GCN propagation computing
                for time_stamp in rating_hash_Tdict_interaction.keys():
                    positive_rating_data_for_item_user[time_stamp] = {}
                    for (u, i) in rating_hash_Tdict_interaction[time_stamp]:
                        if i in positive_rating_data_for_item_user[time_stamp].keys():
                            positive_rating_data_for_item_user[time_stamp][i].append(u)
                        else: 
                            positive_rating_data_for_item_user[time_stamp][i] = []
                            positive_rating_data_for_item_user[time_stamp][i].append(u)

                    item_list = sorted(list(positive_rating_data_for_item_user[time_stamp].keys()))

                '''
                for (u, i) in rating_hash_data_for_item_user:
                    rating_total_data_for_item_user.add((i, u))
                    positive_rating_data_for_item_user[i].add(u)
                '''

                self.positive_rating_data_for_item_user_Tdict = positive_rating_data_for_item_user
                #self.rating_total_data_for_item_user = len(rating_total_data_for_item_user)
                
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'positive_rating_data_for_item_user_Tdict_test.npy',positive_rating_data_for_item_user)

        t_aprdit2 = time()

        print("**** END Arranging Positive Rating Data for Item User, cost %f ****"%(t_aprdit2-t_aprdit1))
       

    def PrintUserInfo(self):
        total_rating_user_list = self.total_rating_user_list
        total_link_user1_list = self.total_link_user1_list

        intersection_user_list = list(set(total_rating_user_list).intersection(set(total_link_user1_list)))


        print("len of rating user:%d"%(len(total_rating_user_list)))
        print("len of link user:%d"%(len(total_link_user1_list)))

        print("max id of rating user:%d"%(max(total_rating_user_list)))
        print("max id of link user:%d"%(max(total_link_user1_list)))

        print("len of intersection user:%d"%(len(intersection_user_list)))
        print("max id of intersection user:%d"%(max(intersection_user_list)))


    def GetLinkMissingPeriod(self):
        total_len = self.conf.total_len
        wholePeriod = []
        for time_ser in range(10):
            wholePeriod.append(time_ser)
        # For link 
        link_user_missing_period_dict = {}
        user_link_stamp_dict = self.user_link_stamp_dict
        for userkey in user_link_stamp_dict:
            link_user_missing_period_dict[userkey] = list(set(wholePeriod).difference(set(user_link_stamp_dict[userkey])))

        self.link_user_missing_period_dict = link_user_missing_period_dict
        #np.save("./data/"+self.conf.data_name+"/"+self.conf.data_name+"_link_user_missing_period_dict",link_user_missing_period_dict)

    def GetRatingMissingPeriod(self):
        total_len = self.conf.total_len
        wholePeriod = []
        for time_ser in range(10):
            wholePeriod.append(time_ser)
        # For link 
        rating_user_missing_period_dict = {}
        user_rating_stamp_dict = self.user_rating_stamp_dict
        for userkey in user_rating_stamp_dict:
            rating_user_missing_period_dict[userkey] = list(set(wholePeriod).difference(set(user_rating_stamp_dict[userkey])))   # need to check

        self.rating_user_missing_period_dict = rating_user_missing_period_dict
        #np.save("./data/"+self.conf.data_name+"/"+self.conf.data_name+"_rating_user_missing_period_dict",rating_user_missing_period_dict)

        


# ==================== Second Part: Data Operation relative  ==================== #
# generating batch, formating data with required input format
    
    
    # Negative items generation for training or test, val
    def generateSocialTrainNegative(self):

        #print("**** Start Sampling Negative Social Data for training ****")
        #t_gstn1 = time()

        social_hash_data_T = self.social_hash_data_T   #(u1,u2,t)  
        social_user_list = self.total_social_user1_list
  
        social_positive_data_for_user1_user2_Tdict = self.social_positive_data_for_user1_user2_Tdict
        social_positive_data_for_user1_user2_T_Udict = self.social_positive_data_for_user1_user2_T_Udict
        social_positive_data_for_user1_user2_withoutR_Tdict = self.social_positive_data_for_user1_user2_withoutR_Tdict
        social_hash_data = self.social_hash_data



        num_users = self.conf.num_users
        num_social_negatives = self.conf.num_social_negatives

        social_negative_data = {}
        total_negative_socialdata = set()
        #hash_data = self.hash_data
        for (u1, u2, t) in social_hash_data_T:
            
            if u1 not in social_negative_data.keys():
                social_negative_data[u1]=[]

            #set_trace()
            for _ in range(num_social_negatives):  # the number of negative social samples
                #print _
                j = np.random.randint(num_users)  # the number of total item

                while (u1,j) in social_hash_data:
                #while j in social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]:
                    #if rating_hash_data[(u,j)]<3   # consider about if the oringinal rating less than 3
                    j = np.random.randint(num_users)
                social_negative_data[u1].append((j, float(0.0), t))
                total_negative_socialdata.add((u1, j, float(0.0), t))

    
        self.terminal_flag = 1
        self.total_negative_socialdata = total_negative_socialdata
        self.social_negative_data = social_negative_data

        #t_gstn2 = time()
        #print("**** End Sampling Negative Social Data for training, cost:%f ****"%(t_gstn2-t_gstn1))

    def generateSocialTestNegative(self):

        social_hash_data_T = self.social_hash_data_T   #(u1,u2,t)  
        social_user_list = self.total_social_user1_list
  
        social_positive_data_for_user1_user2_Tdict = self.social_positive_data_for_user1_user2_Tdict
        social_positive_data_for_user1_user2_T_Udict = self.social_positive_data_for_user1_user2_T_Udict
        social_positive_data_for_user1_user2_withoutR_Tdict = self.social_positive_data_for_user1_user2_withoutR_Tdict
        social_hash_data = self.social_hash_data


        num_users = self.conf.num_users
        num_social_negatives = self.conf.num_social_test_negatives

        social_negative_data = {}
        total_negative_socialdata = set()
        #hash_data = self.hash_data
        for (u1, u2, t) in social_hash_data_T:
            
            if u1 not in social_negative_data.keys():
                social_negative_data[u1]=[]

            #set_trace()
            for _ in range(num_social_negatives):  # the number of negative social samples
                #print _
                j = np.random.randint(num_users)  # the number of total item

                while j in social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]:
                    #if rating_hash_data[(u,j)]<3   # consider about if the oringinal rating less than 3
                    j = np.random.randint(num_users)
                social_negative_data[u1].append((j, float(0.0), t))
                total_negative_socialdata.add((u1, j, float(0.0), t))

    
        self.terminal_flag = 1
        self.total_negative_socialdata = total_negative_socialdata
        self.social_test_negative_data = social_negative_data

    
    # Negative items generation for training or test, val


    def generateRatingTrainNegative(self):
        #print("**** Start Sampling Negative Rating Data for training ****") # cost too much time, and if use RMSE, the rr part should remove
        #t_grtn1 = time()

        if self.conf.num_rating_train_negatives != 0:
            rating_hash_data_T = self.rating_hash_data_T   #(u,v,r,t)  
            rating_user_list = self.total_rating_user
      
            rating_positive_data_for_user_item_Tdict = self.rating_positive_data_for_user_item_Tdict
            rating_positive_data_for_user_item_T_Udict = self.rating_positive_data_for_user_item_T_Udict
            rating_positive_data_for_user_item_withoutR_Tdict = self.rating_positive_data_for_user_item_withoutR_Tdict
            rating_hash_data = self.rating_hash_data


            num_items = self.conf.num_items
            num_rating_negatives = self.conf.num_rating_train_negatives # its 0
            rating_negative_data = {}
            total_negative_ratingdata = set()
            #hash_data = self.hash_data

            
                 
            for (u, i, r, t) in rating_hash_data_T:
                
                if u not in rating_negative_data.keys():
                    rating_negative_data[u]=[]

                #set_trace()
                for _ in range(num_rating_negatives):  # the number of negative samples
                    #print _
                    j = np.random.randint(num_items)  # the number of total item

                    while j in rating_positive_data_for_user_item_withoutR_Tdict[t][u]:
                        #if rating_hash_data[(u,j)]<3   # consider about if the oringinal rating less than 3
                        j = np.random.randint(num_items)
                    rating_negative_data[u].append((j, int(0), t))
                    total_negative_ratingdata.add((u, j, int(0), t))

            
                
                self.total_negative_ratingdata = total_negative_ratingdata
                self.rating_negative_data = rating_negative_data


        self.terminal_flag = 1
        #t_grtn2 = time()
        #print("**** End Sampling Negative Rating Data for training, cost:%f ****"%(t_grtn2-t_grtn1))

    def generateRatingTestNegative(self):
        print("**** Start Sampling Negative Rating Data for test ****")
        t_rtn1 = time()

        rating_hash_data_T = self.rating_hash_data_T   #(u,v,r,t)  
        rating_user_list = self.total_rating_user
  
        rating_positive_data_for_user_item_Tdict = self.rating_positive_data_for_user_item_Tdict
        rating_positive_data_for_user_item_T_Udict = self.rating_positive_data_for_user_item_T_Udict
        rating_positive_data_for_user_item_withoutR_Tdict = self.rating_positive_data_for_user_item_withoutR_Tdict
        rating_hash_data = self.rating_hash_data


        num_items = self.conf.num_items
        num_rating_negatives = self.conf.num_rating_test_negatives
        rating_negative_data = {}
        total_negative_ratingdata = set()
        #hash_data = self.hash_data


        for (u, i, r, t) in rating_hash_data_T:
            
            if u not in rating_negative_data.keys():
                rating_negative_data[u]=[]

            #set_trace()
            for _ in range(num_rating_negatives):  # the number of negative samples
                #print _
                j = np.random.randint(num_items)  # the number of total item

                while j in rating_positive_data_for_user_item_withoutR_Tdict[t][u]:
                    #if rating_hash_data[(u,j)]<3   # consider about if the oringinal rating less than 3
                    j = np.random.randint(num_items)
                rating_negative_data[u].append((j, int(0), t))
                total_negative_ratingdata.add((u, j, int(0), t))

    
        self.terminal_flag = 1
        self.total_negative_ratingdata = total_negative_ratingdata
        self.rating_test_negative_data = rating_negative_data
        t_rtn2 = time()

        #t_snr2 = time()
        print("**** End Sampling Negative Rating Data for test, cost:%f ****"%(t_rtn2-t_rtn1))



    # generate rating training batch 
    def getTrainRatingBatch(self):
        #print("**** Start get train rating batch ****")
        #t_gtrb1 = time()

        rating_positive_data_for_user_item_T_Udict = self.rating_positive_data_for_user_item_T_Udict
        # rating_negative_data = self.rating_negative_data
        rating_user_list = self.rating_user_list   # id up to 4630, but the list len is 4421, means some of the users miss rating records
        index = self.index
        batch_size = self.conf.training_r_batch_size

        user_list, item_list, labels_list, stamp_list = [], [], [], []
        
        if index + batch_size < len(rating_user_list):  # len(rating_user_list) = 4421
            target_user_list = rating_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_user_list = rating_user_list[index:len(rating_user_list)]
            self.index = 0
            self.terminal_flag = 0

        #set_trace()    
        for u in target_user_list:

            user_list.extend([u]*len(rating_positive_data_for_user_item_T_Udict[u]))

            cur_user_list_record = rating_positive_data_for_user_item_T_Udict[u]
            l_cur_user_list_record = len(rating_positive_data_for_user_item_T_Udict[u])
            cur_item_list = []
            cur_label_list = []
            cur_stamp_list = []

            if self.conf.num_rating_train_negatives != 0:
                cur_neg_user_list_record = rating_negative_data[u]
                l_cur_neg_user_list_record = len(rating_negative_data)
                cur_neg_item_list = []
                cur_neg_label_list = []
                cur_neg_stamp_list = []

            #pos
            for record in cur_user_list_record:
                
                item_id = record[0]
                rating_label = record[1]
                stamp = record[2]
                cur_item_list.append(item_id)
                cur_label_list.append(rating_label)
                cur_stamp_list.append(stamp)

            item_list.extend(cur_item_list)
            labels_list.extend(cur_label_list)
            stamp_list.extend(cur_stamp_list)

            #set_trace()

            
            if self.conf.num_rating_train_negatives != 0:
           
                user_list.extend([u]*len(rating_negative_data[u]))

                for record in cur_neg_user_list_record:

                    item_id = record[0]
                    rating_label = record[1]
                    stamp = record[2]
                    cur_neg_item_list.append(item_id)
                    cur_neg_label_list.append(rating_label)
                    cur_neg_stamp_list.append(stamp)


                item_list.extend(cur_neg_item_list)
                labels_list.extend(cur_neg_label_list)
                stamp_list.extend(cur_neg_stamp_list)


        self.b_rating_user_list = np.reshape(user_list, [-1, 1])
        self.b_rating_item_list = np.reshape(item_list, [-1, 1])
        self.b_rating_labels_list = np.reshape(labels_list, [-1, 1])        
        self.b_rating_stamp_list = np.reshape(stamp_list,[-1,1])

        #t_gtrb2 = time()
        #print("**** End get train rating batch, cost:%f ****"%(t_gtrb2-t_gtrb1))

    # generate social training batch 
    def getTrainSocialBatch(self):
        #print("**** Start get train social batch ****")
        #t_gtsb1 = time()

        social_positive_data_for_user1_user2_T_Udict = self.social_positive_data_for_user1_user2_T_Udict
        social_negative_data = self.social_negative_data
        social_user_list = self.total_social_user1_list   # id up to 4629(4629), but the list len is 4421(4381), means some of the users miss rating(social) records
        index = self.index
        batch_size = self.conf.training_s_batch_size

        user1_list, user2_list, labels_list, stamp_list = [], [], [], []
        
        if index + batch_size < len(social_user_list):  # len(rating_user_list) = 4421
            target_user_list = social_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_user_list = social_user_list[index:len(social_user_list)]
            self.index = 0
            self.terminal_flag = 0

        #set_trace()    
        for u1 in target_user_list:

            user1_list.extend([u1]*len(social_positive_data_for_user1_user2_T_Udict[u1]))

            cur_user_list_record = social_positive_data_for_user1_user2_T_Udict[u1]
            l_cur_user_list_record = len(social_positive_data_for_user1_user2_T_Udict[u1])
            cur_u2_list = []
            cur_label_list = []
            cur_stamp_list = []

            cur_neg_user_list_record = social_negative_data[u1]
            l_cur_neg_user_list_record = len(social_negative_data)
            cur_neg_u2_list = []
            cur_neg_label_list = []
            cur_neg_stamp_list = []

            #pos
            for record in cur_user_list_record:
                
                u2_id = record[0]
                relation_label = record[1]
                stamp = record[2]
                cur_u2_list.append(u2_id)
                cur_label_list.append(relation_label)
                cur_stamp_list.append(stamp)

            user2_list.extend(cur_u2_list)
            labels_list.extend(cur_label_list)
            stamp_list.extend(cur_stamp_list)


            #neg
            user1_list.extend([u1]*len(social_negative_data[u1]))

            for record in cur_neg_user_list_record:

                u2_id = record[0]
                relation_label = record[1]
                stamp = record[2]
                cur_neg_u2_list.append(u2_id)
                cur_neg_label_list.append(relation_label)
                cur_neg_stamp_list.append(stamp)


            user2_list.extend(cur_neg_u2_list)
            labels_list.extend(cur_neg_label_list)
            stamp_list.extend(cur_neg_stamp_list)


        self.b_social_user1_list = np.reshape(user1_list, [-1, 1])
        self.b_social_user2_list = np.reshape(user2_list, [-1, 1])
        self.b_social_labels_list = np.reshape(labels_list, [-1, 1])        
        self.b_social_stamp_list = np.reshape(stamp_list,[-1,1])

        #t_gtsb2 = time()
        #print("**** End get train social batch, cost:%f ****"%(t_gtsb2-t_gtsb1))



    ''' ###################################################################################################
        EVALUATION
    ''' ###################################################################################################

    # one batch, no labels for eva testing, calculate the positve samples 
    # Rating
    def getEvaRatingPositiveBatchOneTime(self):
        #print("**** Start get eva rating positive batch one time ****")
        #t_gerb1 = time()

        rating_hash_data_T = self.rating_hash_data_T
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i, r, t) in rating_hash_data_T:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        self.eva_rating_index_dict = index_dict

        #t_gerb2 = time()
        #print("**** End get eva rating positive batch one time, cost:%f ****"%(t_gerb2-t_gerb1))
        


    # Social
    def getEvaSocialPositiveBatchOneTime(self):
        #print("**** Start get eva soical positive batch one time ****")
        #t_gesb1 = time()
        social_hash_data_T = self.social_hash_data_T
        user1_list = []
        user2_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u1, u2, t) in social_hash_data_T:
            user1_list.append(u1)
            user2_list.append(u2)
            index_dict[u1].append(index)
            index = index + 1
        self.eva_user1_list = np.reshape(user1_list, [-1, 1])
        self.eva_user2_list = np.reshape(user2_list, [-1, 1])
        self.eva_social_index_dict = index_dict
        #t_gesb2 = time()
        #print("**** End get eva soical positive batch one time, cost:%f ****"%(t_gesb2-t_gesb1))

    # Rating
    def getOneBatchForRatingTestLoss(self):

        rating_positive_data_for_user_item_T_Udict = self.rating_positive_data_for_user_item_T_Udict
        rating_negative_data = self.rating_test_negative_data
        rating_user_list = self.rating_user_list   # id up to 4630, but the list len is 4421, means some of the users miss rating records
        index = self.index
        batch_size = self.conf.training_r_batch_size

        #positive_data = self.positive_data
        user_list, item_list, labels_list, stamp_list = [], [], [], []


        for u in rating_user_list:

            user_list.extend([u]*len(rating_positive_data_for_user_item_T_Udict[u]))
            cur_user_list_record = rating_positive_data_for_user_item_T_Udict[u]
            l_cur_user_list_record = len(rating_positive_data_for_user_item_T_Udict[u])
            cur_item_list = []
            cur_label_list = []
            cur_stamp_list = []

            cur_neg_user_list_record = rating_negative_data[u]
            l_cur_neg_user_list_record = len(rating_negative_data)
            cur_neg_item_list = []
            cur_neg_label_list = []
            cur_neg_stamp_list = []

            #pos
            for record in cur_user_list_record:
                
                item_id = record[0]
                rating_label = record[1]
                stamp = record[2]
                cur_item_list.append(item_id)
                cur_label_list.append(rating_label)
                cur_stamp_list.append(stamp)

            item_list.extend(cur_item_list)
            labels_list.extend(cur_label_list)
            stamp_list.extend(cur_stamp_list)


            #neg
            user_list.extend([u]*len(rating_negative_data[u]))

            for record in cur_neg_user_list_record:
                item_id = record[0]
                rating_label = record[1]
                stamp = record[2]
                cur_neg_item_list.append(item_id)
                cur_neg_label_list.append(rating_label)
                cur_neg_stamp_list.append(stamp)


            item_list.extend(cur_neg_item_list)
            labels_list.extend(cur_neg_label_list)
            stamp_list.extend(cur_neg_stamp_list)


        self.test_rating_user_list = np.reshape(user_list, [-1, 1])
        self.test_rating_item_list = np.reshape(item_list, [-1, 1])
        self.test_rating_labels_list = np.reshape(labels_list, [-1, 1])  


    # Social
    def getOneBatchForSocialTestLoss(self):
        #print("**** Start get one batch for social test loss ****")
        #t_gobst1 = time()


        social_positive_data_for_user1_user2_T_Udict = self.social_positive_data_for_user1_user2_T_Udict
        social_negative_data = self.social_test_negative_data
        social_user_list = self.total_social_user1_list   # id up to 4630, but the list len is 4421, means some of the users miss rating records
        index = self.index
        batch_size = self.conf.training_s_batch_size

        #positive_data = self.positive_data
        user1_list, user2_list, labels_list, stamp_list = [], [], [], []


        for u1 in social_user_list:

            user1_list.extend([u1]*len(social_positive_data_for_user1_user2_T_Udict[u1]))
            cur_user_list_record = social_positive_data_for_user1_user2_T_Udict[u1]
            l_cur_user_list_record = len(social_positive_data_for_user1_user2_T_Udict[u1])
            cur_u2_list = []
            cur_label_list = []
            cur_stamp_list = []

            cur_neg_user_list_record = social_negative_data[u1]
            l_cur_neg_user_list_record = len(social_negative_data)
            cur_neg_u2_list = []
            cur_neg_label_list = []
            cur_neg_stamp_list = []

            #pos
            for record in cur_user_list_record:
                
                u2_id = record[0]
                relation_label = record[1]
                stamp = record[2]
                cur_u2_list.append(u2_id)
                cur_label_list.append(relation_label)
                cur_stamp_list.append(stamp)

            user2_list.extend(cur_u2_list)
            labels_list.extend(cur_label_list)
            stamp_list.extend(cur_stamp_list)


            #neg
            user1_list.extend([u1]*len(social_negative_data[u1]))

            for record in cur_neg_user_list_record:
                u2_id = record[0]
                relation_label = record[1]
                stamp = record[2]
                cur_neg_u2_list.append(u2_id)
                cur_neg_label_list.append(relation_label)
                cur_neg_stamp_list.append(stamp)


            user2_list.extend(cur_neg_u2_list)
            labels_list.extend(cur_neg_label_list)
            stamp_list.extend(cur_neg_stamp_list)


        self.test_social_user1_list = np.reshape(user1_list, [-1, 1])
        self.test_social_user2_list = np.reshape(user2_list, [-1, 1])
        self.test_social_labels_list = np.reshape(labels_list, [-1, 1])              
        

        #t_gobst2 = time()
        #print("**** End get one batch for social test loss, cost:%f ****"%(t_gobst2-t_gobst1))

    '''
        This function designes for the negative data generation process in rating evaluate part
    '''
    # Rating
    def generateEvaRatingNegative(self):
        print("**** Start generate eva rating negtives ****")
        t_gevarn1 = time()
        #rating_user_list = self.rating_user_list
        rating_user_list = self.total_rating_user
        rating_hash_data = self.rating_hash_data
        rating_hash_data_T = self.rating_hash_data_T   #(u,v,r,t)  
      
        num_evaluate = self.conf.num_evaluate
        num_items = self.conf.num_items
        eva_negative_rating_data = defaultdict(list)
        #user_list = sorted(list(rating_positive_data_for_user_item_T_Udict.keys()))

        for u in rating_user_list:   # iteration for user


            for _ in range(num_evaluate):
                j = np.random.randint(num_items)
                while (u, j) in rating_hash_data:
                    j = np.random.randint(num_items)
                eva_negative_rating_data[u].append(j)

        self.eva_negative_rating_data = eva_negative_rating_data

        t_gevarn2 = time()
        print("**** End generate eva rating negtives, cost:%f ****"%(t_gevarn2-t_gevarn1))


    # Social
    def generateEvaSocialNegative(self,train_hash_data):

        print("**** Start generate eva social negtives ****")
        t_gevasn1 = time()
        social_user_list = self.total_social_user1_list
        social_eva_hash_data = self.social_hash_data
        social_hash_data = train_hash_data
      
        num_evaluate = self.conf.num_social_evaluate
        num_users = self.conf.num_users
        eva_negative_social_data = defaultdict(list)
        #set_trace()
        #user_list = sorted(list(rating_positive_data_for_user_item_T_Udict.keys()))

        #set_trace()
        for u1 in social_user_list:   # iteration for user

            for _ in range(num_evaluate):
                j = np.random.randint(num_users) 
                #((1522,2941) in social_eva_hash_data) | ((27,3341) in social_hash_data)
                while (((u1,j) in social_eva_hash_data) | ((u1,j) in social_hash_data)):  #100 evaluate sample not in train or test
                    j = np.random.randint(num_users)
                eva_negative_social_data[u1].append(j)

        self.eva_negative_social_data = eva_negative_social_data
        t_gevasn2 = time()
        print("**** End generate eva social negtives, cost:%f ****"%(t_gevasn2-t_gevasn1))



    '''
        This function designs for the rating evaluate section, generate negative batch
    '''

    #Rating
    def getEvaRatingRankingBatch(self):
        batch_size = self.conf.evaluate_batch_size
        num_evaluate = self.conf.num_evaluate
        eva_negative_rating_data = self.eva_negative_rating_data
        rating_user_list = self.total_rating_user
        index = self.index
        terminal_flag = 1
        total_users = len(rating_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = rating_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = rating_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u]*num_evaluate)
            item_list.extend(eva_negative_rating_data[u])
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        return batch_user_list, terminal_flag

    #Social
    def getEvaSocialRankingBatch(self):
        batch_size = self.conf.evaluate_batch_size
        num_social_evaluate = self.conf.num_social_evaluate
        eva_negative_social_data = self.eva_negative_social_data
        social_user_list = self.total_social_user1_list
        index = self.index
        terminal_flag = 1
        total_users = len(social_user_list)
        user1_list = []
        user2_list = []
        if index + batch_size < total_users:
            batch_user_list = social_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = social_user_list[index:total_users]
            self.index = 0
        for u1 in batch_user_list:
            user1_list.extend([u1]*num_social_evaluate)
            user2_list.extend(eva_negative_social_data[u1])
        self.eva_user1_list = np.reshape(user1_list, [-1, 1])
        self.eva_user2_list = np.reshape(user2_list, [-1, 1])
        return batch_user_list, terminal_flag




    def generateConsumedItemsSparseMatrix(self):

        print("**** START Generate Consumed Items Sparse Matrix ****")

        try: 

            print("**** START Loading Consumed Items Sparse Matrix ****")
            t1=time()
            self.consumed_items_indices_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'consumed_items_indices_Tdict_list.npy',allow_pickle=True).item()
            self.consumed_items_values_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'consumed_items_values_Tdict_list.npy',allow_pickle=True).item()  
            t2=time()
            print("**** End Loading Consumed Items Sparse Matrix, cost:%f ****"%(t2-t1))

        except IOError:
            print("**** No current files, START creating Consumed Items Sparse Matrix ****")
            t_gcisM1 = time()

            rating_positive_data_for_user_item_T_Udict = self.rating_positive_data_for_user_item_T_Udict  
            rating_positive_data_for_user_item_withoutR_Tdict = self.rating_positive_data_for_user_item_withoutR_Tdict

            #**** old version ****#
            '''
            consumed_items_indices_list = []
            consumed_items_values_list = []
            consumed_item_num_list = []
            consumed_items_dict = defaultdict(list)
            '''


            consumed_items_indices_Tdict_list = {}
            consumed_items_values_Tdict_list = {}
            consumed_item_num_Tdict_list = {}
            consumed_items_T_dict = {}


            #******** New version for Temporal Sparse data Format ********#

            user_list = {}
            for t in rating_positive_data_for_user_item_withoutR_Tdict.keys():
                consumed_items_T_dict[t] = {}
                for u in rating_positive_data_for_user_item_withoutR_Tdict[t].keys():
                    consumed_items_T_dict[t][u] = sorted(rating_positive_data_for_user_item_withoutR_Tdict[t][u])   # whether work??

                user_list[t]=sorted(list(rating_positive_data_for_user_item_withoutR_Tdict[t].keys()))



            for t in rating_positive_data_for_user_item_withoutR_Tdict.keys():
                consumed_items_indices_Tdict_list[t] = []
                consumed_items_values_Tdict_list[t] = []
                for u in user_list[t]:
                    if u==3677:
                        flag=1
                     
                    for i in rating_positive_data_for_user_item_withoutR_Tdict[t][u]:
                        consumed_items_indices_Tdict_list[t].append([u,i])
                        consumed_items_values_Tdict_list[t].append(1.0/len(consumed_items_T_dict[t][u]))


          
            self.consumed_items_indices_Tdict_list = {}
            self.consumed_items_values_Tdict_list = {}
            for t in rating_positive_data_for_user_item_withoutR_Tdict.keys():
                self.consumed_items_indices_Tdict_list[t] = np.array(consumed_items_indices_Tdict_list[t]).astype(np.int64)
                self.consumed_items_values_Tdict_list[t] = np.array(consumed_items_values_Tdict_list[t]).astype(np.float32)

            
            np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'consumed_items_indices_Tdict_list.npy',self.consumed_items_indices_Tdict_list)
            np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'consumed_items_values_Tdict_list.npy',self.consumed_items_values_Tdict_list)
            
            t_gcisM2 = time()
            print("**** END creating Consumed Items Sparse Matrix(Finish), cost: %f ****"%(t_gcisM2-t_gcisM1))     # 53s, 读取可以优化
  

    

    def generateConsumedItemsSparseMatrixForItemUser(self):
        t_gcism1 = time()
        try:
            self.item_customer_values_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'item_customer_values_Tdict_list.npy',allow_pickle=True).item()
            self.item_customer_indices_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'item_customer_indices_Tdict_list.npy',allow_pickle=True).item()
        except IOError:
            
            print("**** START Generate Item Customer Sparse Matrix ****")
            
            #positive_data_for_item_user = self.positive_data_for_item_user  
            positive_rating_data_for_item_user_Tdict = self.positive_rating_data_for_item_user_Tdict
            #set_trace()

            item_customer_indices_Tdict_list = {}
            item_customer_values_Tdict_list = {}



            #item_customer_values_weight_avg_list = []
            #item_customer_num_list = []
            item_customer_Tdict = defaultdict(list)
            
            item_list = {}
            for t in positive_rating_data_for_item_user_Tdict.keys():
                item_customer_Tdict[t]={}
                for i in positive_rating_data_for_item_user_Tdict[t].keys():
                    item_customer_Tdict[t][i] = sorted(list(positive_rating_data_for_item_user_Tdict[t][i]))

                item_list[t] = sorted(list(positive_rating_data_for_item_user_Tdict[t]))

          

            for t in positive_rating_data_for_item_user_Tdict.keys():
                item_customer_indices_Tdict_list[t] = []
                item_customer_values_Tdict_list[t] = []
                for i in item_list[t]:
                    for u in positive_rating_data_for_item_user_Tdict[t][i]:
                        item_customer_indices_Tdict_list[t].append([i,u])
                        item_customer_values_Tdict_list[t].append(1.0/len(item_customer_Tdict[t][i]))


            self.item_customer_indices_Tdict_list = {}
            self.item_customer_values_Tdict_list = {}
            for t in positive_rating_data_for_item_user_Tdict.keys():        
                self.item_customer_indices_Tdict_list[t] = np.array(item_customer_indices_Tdict_list[t]).astype(np.int64)
                self.item_customer_values_Tdict_list[t] = np.array(item_customer_values_Tdict_list[t]).astype(np.float32)


            np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'item_customer_indices_Tdict_list.npy',item_customer_indices_Tdict_list)
            np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'item_customer_values_Tdict_list.npy',item_customer_values_Tdict_list)


        t_gcism2 = time()
        print("**** END Generate Items Customer Sparse Matrix, cost: %f ****"%(t_gcism2-t_gcism1))



    def generateSocialNeighborsSparseMatrix(self):
            #social_neighbors = self.social_neighbors
 

            social_positive_data_for_user1_user2_T_Udict = self.social_positive_data_for_user1_user2_T_Udict  
            social_positive_data_for_user1_user2_withoutR_Tdict = self.social_positive_data_for_user1_user2_withoutR_Tdict


            consumed_items_indices_Tdict_list = {}
            consumed_items_values_Tdict_list = {}
            consumed_item_num_Tdict_list = {}
            consumed_items_T_dict = {}

            #set_trace()
            social_neighbors_indices_Tdict_list = {}
            social_neighbors_values_Tdict_list = {}
            social_neighbor_num_Tdict_list = {}
            social_neighbors_T_dict = defaultdict(list)

            user_user_num_for_sparsity_dict = defaultdict(set)
            user_user_sparsity_dict = {}


            social_user_list = {}
            for t in social_positive_data_for_user1_user2_withoutR_Tdict.keys():
                social_neighbors_T_dict[t] = defaultdict(list)
                for u1 in social_positive_data_for_user1_user2_withoutR_Tdict[t].keys():
                    social_neighbors_T_dict[t][u1] = sorted(list(social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]))

                social_user_list[t] =  sorted(list(social_positive_data_for_user1_user2_withoutR_Tdict[t].keys()))

            self.social_neighbors_T_dict = social_neighbors_T_dict


            
            
            for t in social_positive_data_for_user1_user2_withoutR_Tdict.keys():

                social_neighbors_indices_Tdict_list[t] = []
                social_neighbors_values_Tdict_list[t] = []
                for u1 in social_user_list[t]:
                    for u2 in social_positive_data_for_user1_user2_withoutR_Tdict[t][u1]:
                        social_neighbors_indices_Tdict_list[t].append([u1,u2])
                        social_neighbors_values_Tdict_list[t].append(1.0/len(social_neighbors_T_dict[t][u1]))

            self.social_neighbors_indices_Tdict_list = {}
            self.social_neighbors_values_Tdict_list = {}
            #set_trace()
            for t in social_positive_data_for_user1_user2_withoutR_Tdict.keys():
                self.social_neighbors_indices_Tdict_list[t] = np.array(social_neighbors_indices_Tdict_list[t]).astype(np.int64)
                self.social_neighbors_values_Tdict_list[t] = np.array(social_neighbors_values_Tdict_list[t]).astype(np.float32)




################################ Balance Social Neighbors ################################

    def arrangeBalanceSocialData(self):
        
            social_neighbors_num_graph_Tdict = self.social_neighbors_T_dict # first trust set
            social_user_list = self.social_neighbors_T_dict
            num_users = self.conf.num_users
            k = self.conf.k
            k_2order_negtive = self.conf.k_2order_negtive
            num_limit = self.conf.num_limit
            
            
            # First distrust set
            try:
                print("**** START Loading 1 hop arrangeBalanceSocialData")
                t_absd1 = time()
                self.final_first_distrust_neighbors_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data/%s_factor/'%k+'First_distrust_neighbors_Tdict_current.npy' ,allow_pickle=True).item()
                self.all_distrust_neighbors_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/'+'All_distrust_neighbors_Tdict_train.npy',allow_pickle=True).item()
                t_absd2 = time()
                print("**** END Loading 1 hop arrangeBalanceSocialData, cost: %f ****"%(t_absd2-t_absd1))

            except IOError:
                
                print("**** START create 1-hop relationships of function arrangeBalanceSocialData")
                t_absd1_1hop = time()


                distrust_neighbors_Tdict = defaultdict(set)
                final_first_distrust_neighbors_Tdict = defaultdict(set)

                # Extract all untrusted users
                def generate_distrust_neighbors(user):
                    return (other_user for other_user in range(num_users) if other_user not in social_neighbors_num_graph_Tdict[t][user] and other_user != user)
                
                for t in social_user_list.keys():
                    distrust_neighbors_Tdict[t] = defaultdict(set)
                    final_first_distrust_neighbors_Tdict[t] = defaultdict(set)

                    for user in range(num_users):
                        distrust_neighbors_gen = generate_distrust_neighbors(user)
                        # Convert the generator to a list and save it to a dictionary
                        distrust_neighbors_Tdict[t][user] = set(distrust_neighbors_gen)
                        
                        # Do a random sampling of first-order distrusted users
                        num_trust = len(social_neighbors_num_graph_Tdict[t][user])
                        if (k+1) * num_trust > num_users:
                            num_distrust = num_users - num_trust -1 #If it exceeds the number that can be drawn, draw all negative samples directly (removing itself)
                        else:
                            num_distrust = k * num_trust #Not exceeded, k-fold positive sample size drawn

                        # Randomly select a user from the set of untrusted users for that user
                        final_distrust_neighbors = random.sample(list(distrust_neighbors_Tdict[t][user]), num_distrust)
                        
                        # Save to Dictionary
                        final_first_distrust_neighbors_Tdict[t][user] = sorted(set(final_distrust_neighbors))

                self.all_distrust_neighbors_Tdict = distrust_neighbors_Tdict
                self.final_first_distrust_neighbors_Tdict = final_first_distrust_neighbors_Tdict
                # set_trace()
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/'+'All_distrust_neighbors_Tdict_train.npy',distrust_neighbors_Tdict)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data/%s_factor/'%k+'First_distrust_neighbors_Tdict_current.npy' ,final_first_distrust_neighbors_Tdict)

                t_absd2_2hop = time()
                print("**** END create 1-hop relationships of function, cost: %f ****"%(t_absd2_2hop-t_absd1_1hop))


            
            # Second trust set
            try:
                print("**** START Loading 2hop trust arrangeBalanceSocialData") 
                t_absd1 = time()
                self.first_trust_second_trust_neighbors_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'First_trust_Second_trust_neighbors_%sneg_Tdict.npy'%k_2order_negtive,allow_pickle=True).item()
                self.first_dis_second_dis_neighbors_Tdict = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'First_dis_Second_dis_neighbors_%sneg_Tdict.npy'%k_2order_negtive,allow_pickle=True).item()
                t_absd2 = time()
                print("**** END Loading  2hop  trust arrangeBalanceSocialData, cost: %f ****"%(t_absd2-t_absd1))
            
            except IOError:

                print("**** START create 2-hop trust relationships of function arrangeBalanceSocialData")
                t_absd1_1hop = time()          

                first_trust_second_trust_neighbors = defaultdict(set)
                first_dis_second_dis_neighbors = defaultdict(set)
                second_trust_neighbors = defaultdict(set)

                #（+ +）
                for t in social_user_list.keys():
                    first_trust_second_trust_neighbors[t] = defaultdict(set)
                    for user in range(num_users):
                        # For each user, extract the trust users they believe in as a second-order positive relationship
                        for trust_neighbor in social_neighbors_num_graph_Tdict[t][user]:
                            first_trust_second_trust_neighbors[t][user].update(social_neighbors_num_graph_Tdict[t][trust_neighbor])

                #（- -）
                for t in social_user_list.keys():
                    first_dis_second_dis_neighbors[t] = defaultdict(set)
                    for user in range(num_users):
                        # Check the number of first-order negative relationships
                        num_first_dis = len(self.final_first_distrust_neighbors_Tdict[t][user])
                        if num_first_dis <= num_limit: #Not exceeding the quantity limit
                            for distrust_neighbor in self.final_first_distrust_neighbors_Tdict[t][user]:
                                num_extract = k_2order_negtive
                                # Randomly select （num_extract） from second-order negative relationships
                                tmp_extract_distrust_neighbors = random.sample(list(self.all_distrust_neighbors_Tdict[t][distrust_neighbor]), num_extract)
                                first_dis_second_dis_neighbors[t][user].update(set(tmp_extract_distrust_neighbors))
                        else: # Exceed the quantity limit
                            tmp_neighbors_list = random.sample(list(self.final_first_distrust_neighbors_Tdict[t][user]), num_limit)
                            for distrust_neighbor in tmp_neighbors_list:
                                num_extract = k_2order_negtive
                                tmp_extract_distrust_neighbors = random.sample(list(self.all_distrust_neighbors_Tdict[t][distrust_neighbor]), num_extract)
                                first_dis_second_dis_neighbors[t][user].update(set(tmp_extract_distrust_neighbors))

                
                #（+ +）and（- -）
                for t in social_user_list.keys():
                    second_trust_neighbors[t] = defaultdict(set)
                    for user in range(num_users):
                        second_trust_neighbors[t][user] = set(first_trust_second_trust_neighbors[t][user].union(first_dis_second_dis_neighbors[t][user]))
                
                self.first_trust_second_trust_neighbors_Tdict = first_trust_second_trust_neighbors
                self.first_dis_second_dis_neighbors_Tdict = first_dis_second_dis_neighbors
                self.final_second_trust_neighbors_Tdict = second_trust_neighbors

                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'First_trust_Second_trust_neighbors_%sneg_Tdict.npy'%k_2order_negtive, first_trust_second_trust_neighbors)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'First_dis_Second_dis_neighbors_%sneg_Tdict.npy'%k_2order_negtive, first_dis_second_dis_neighbors)
                # np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'Second_trust_neighbors_%sneg_Tdict.npy'%k_2order_negtive, second_trust_neighbors)

                t_absd2_2hop = time()
                print("**** END create 2-hop trust relationships of function, cost: %f ****"%(t_absd2_2hop-t_absd1_1hop))
            


    
    def generateBalanceSocialNeighborsSparseMatrix(self): 
            t_gbsnsm1 = time()
            k = self.conf.k
            k_2order_negtive = self.conf.k_2order_negtive
            print("**** START Generate Balance Social Neighbors Sparse Matrix ****")

            # (++)
            try:

                print("**** START loading second trust Balance Social Neighbors Sparse Matrix 2.1hop ****")
                t1=time()

                self.first_trust_second_trust_neighbors_Tdict_indices_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_trust_second_trust_neighbors_Tdict_indices_Tdict_list_%sneg.npy'%k_2order_negtive, allow_pickle=True).item()
                self.first_trust_second_trust_neighbors_Tdict_values_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_trust_second_trust_neighbors_Tdict_values_Tdict_list_%sneg.npy'%k_2order_negtive, allow_pickle=True).item()
            
                t2 =time()
                print("END Loading second trust Balance Social Neighbors Sparse Matrix 2.1hop, cost: %f"%(t2-t1))

            except IOError:

                print("**** START creating second trust Balance Social Neighbors Sparse Matrix 2.1hop ****")
                t1=time()
            
                #Generate the indices and values of the SparseMatrix of the (++) set
                first_trust_second_trust_neighbors_Tdict = self.first_trust_second_trust_neighbors_Tdict

                first_trust_second_trust_neighbors_Tdict_indices_Tdict_list = {}
                first_trust_second_trust_neighbors_Tdict_values_Tdict_list = {}

                for stamp in first_trust_second_trust_neighbors_Tdict.keys():
                    first_trust_second_trust_neighbors_Tdict_indices_Tdict_list[stamp] = []
                    first_trust_second_trust_neighbors_Tdict_values_Tdict_list[stamp] = []

                    for u1 in first_trust_second_trust_neighbors_Tdict[stamp].keys():
                        tmp_neighbors_list = sorted(list(first_trust_second_trust_neighbors_Tdict[stamp][u1]))
                        
                        for u2 in tmp_neighbors_list:
                            
                            # indices
                            first_trust_second_trust_neighbors_Tdict_indices_Tdict_list[stamp].append([u1,u2])
                            # values_avg
                            first_trust_second_trust_neighbors_Tdict_values_Tdict_list[stamp].append(1.0/len(first_trust_second_trust_neighbors_Tdict[stamp][u1]))

                self.first_trust_second_trust_neighbors_Tdict_indices_Tdict_list = first_trust_second_trust_neighbors_Tdict_indices_Tdict_list
                self.first_trust_second_trust_neighbors_Tdict_values_Tdict_list = first_trust_second_trust_neighbors_Tdict_values_Tdict_list
                
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_trust_second_trust_neighbors_Tdict_indices_Tdict_list_%sneg.npy'%k_2order_negtive, first_trust_second_trust_neighbors_Tdict_indices_Tdict_list)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_trust_second_trust_neighbors_Tdict_values_Tdict_list_%sneg.npy'%k_2order_negtive, first_trust_second_trust_neighbors_Tdict_values_Tdict_list)
                # set_trace()

                t2 =time()
                print("END creating second trust Balance Social Neighbors Sparse Matrix 2.1hop, cost: %f"%(t2-t1))


            # (--)
            try:

                print("**** START loading second trust Balance Social Neighbors Sparse Matrix 2.2hop ****")
                t1=time()

                self.first_dis_second_dis_neighbors_Tdict_indices_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_dis_second_dis_neighbors_Tdict_indices_Tdict_list_%sneg.npy'%k_2order_negtive, allow_pickle=True).item()
                self.first_dis_second_dis_neighbors_Tdict_values_Tdict_list = np.load(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_dis_second_dis_neighbors_Tdict_values_Tdict_list_%sneg.npy'%k_2order_negtive, allow_pickle=True).item()
            
                t2 =time()
                print("END Loading second trust Balance Social Neighbors Sparse Matrix 2.2hop, cost: %f"%(t2-t1))

            except IOError:

                print("**** START creating second trust Balance Social Neighbors Sparse Matrix 2.2hop ****")
                t1=time()
            
                #Generate the indices and values of the SparseMatrix of the (--) set
                first_dis_second_dis_neighbors_Tdict = self.first_dis_second_dis_neighbors_Tdict

                first_dis_second_dis_neighbors_Tdict_indices_Tdict_list = {}
                first_dis_second_dis_neighbors_Tdict_values_Tdict_list = {}

                for stamp in first_dis_second_dis_neighbors_Tdict.keys():
                    first_dis_second_dis_neighbors_Tdict_indices_Tdict_list[stamp] = []
                    first_dis_second_dis_neighbors_Tdict_values_Tdict_list[stamp] = []

                    for u1 in first_dis_second_dis_neighbors_Tdict[stamp].keys():
                        tmp_neighbors_list = sorted(list(first_dis_second_dis_neighbors_Tdict[stamp][u1]))
                        
                        for u2 in tmp_neighbors_list:
                            # indices
                            first_dis_second_dis_neighbors_Tdict_indices_Tdict_list[stamp].append([u1,u2])
                            # values_avg
                            first_dis_second_dis_neighbors_Tdict_values_Tdict_list[stamp].append(1.0/len(first_dis_second_dis_neighbors_Tdict[stamp][u1]))

                self.first_dis_second_dis_neighbors_Tdict_indices_Tdict_list = first_dis_second_dis_neighbors_Tdict_indices_Tdict_list
                self.first_dis_second_dis_neighbors_Tdict_values_Tdict_list = first_dis_second_dis_neighbors_Tdict_values_Tdict_list
                
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_dis_second_dis_neighbors_Tdict_indices_Tdict_list_%sneg.npy'%k_2order_negtive, first_dis_second_dis_neighbors_Tdict_indices_Tdict_list)
                np.save(os.getcwd()+'/data/'+self.conf.data_name+'/balance_unbalance_data_new/%s_factor/'%k+'first_dis_second_dis_neighbors_Tdict_values_Tdict_list_%sneg.npy'%k_2order_negtive, first_dis_second_dis_neighbors_Tdict_values_Tdict_list)

                t2 =time()
                print("END creating second trust Balance Social Neighbors Sparse Matrix 2.2hop, cost: %f"%(t2-t1))
