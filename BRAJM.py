#coding=utf-8
from __future__ import division
import tensorflow as tf
import numpy as np
from ipdb import set_trace
from time import time
from tensorflow.contrib.rnn.python.ops import rnn_cell
#import keras

class DJBM():
    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'ITEM_CUSTOMER_SPARSE_MATRIX'
        )

    def startConstructGraph(self):
        t_1=time()
        self.initializeNodes()
        t_2=time()
        self.constructTrainGraph()
        t_3=time()
        self.saveVariables()
        t_4=time()
        self.defineMap()
        t_5=time()

        print("cost:%f,%f,%f,%f"%(t_2-t_1, t_3-t_1, t_4-t_1, t_5-t_1))



    def inputSupply(self, data_dict):
        low_att_std = 1.0

        #---------------- USER-USER ----------------#

        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']

        self.social_neighbors_values_input1 = {}
        first_mean_social_influ_list = []
        first_var_social_influ_list = []

        self.social_neighbors_values_input2 = {}
        self.social_neighbors_values_input3 = {}

        self.first_low_att_layer_for_social_neighbors_layer1 = {}
        self.first_low_att_layer_for_social_neighbors_layer2 = {}

        self.second_low_att_layer_for_social_neighbors_layer1 = {}
        self.second_low_att_layer_for_social_neighbors_layer2 = {}

        self.third_low_att_layer_for_social_neighbors_layer1 = {}
        self.third_low_att_layer_for_social_neighbors_layer2 = {}

        for t in range(self.conf.total_len):# 0-10
            self.first_low_att_layer_for_social_neighbors_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_SN_layer1_' + str(t))
            self.first_low_att_layer_for_social_neighbors_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_SN_layer2_' + str(t))

            self.second_low_att_layer_for_social_neighbors_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_SN_layer1_' + str(t))
            self.second_low_att_layer_for_social_neighbors_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_SN_layer2_' + str(t))
            
            self.third_low_att_layer_for_social_neighbors_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_SN_layer1_' + str(t))
            self.third_low_att_layer_for_social_neighbors_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='third_low_att_SN_layer2_' + str(t))
            
        for t in range(self.conf.total_len):# 0-10
            #set_trace()
            #1st_layer
            self.social_neighbors_values_input1[t] = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_social_neighbors_layer2[t](self.first_low_att_layer_for_social_neighbors_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input[t])], stddev=low_att_std)),[-1,1])      ) )  ),1)
            first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_values_input1[t], axes=0)

            first_mean_social_influ_list.append(first_mean_social_influ)
            first_var_social_influ_list.append(first_var_social_influ)     

            # 2nd_layer
            self.social_neighbors_values_input2[t] = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_social_neighbors_layer2[t](self.second_low_att_layer_for_social_neighbors_layer1[t]( \
                                tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input[t])], stddev=low_att_std)),[-1,1])      ) )  ),1)
            
            # 3rd_layer
            self.social_neighbors_values_input3[t] = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_social_neighbors_layer2[t](self.third_low_att_layer_for_social_neighbors_layer1[t]( \
                                tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input[t])], stddev=low_att_std)),[-1,1])      ) )  ),1)        
            

        #first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_values_input1, axes=0)
        self.first_user_user_low_att = [first_mean_social_influ_list, first_var_social_influ_list]



        #---------------- USER-USER-Balance ----------------#

        self.first_trust_second_trust_neighbors_indices_input = data_dict['FIRST_TRUST_SECOND_TRUST_NEIGHBORS_INDICES_INPUT']
        self.first_trust_second_trust_neighbors_values_input_avg = data_dict['FIRST_TRUST_SECOND_TRUST_NEIGHBORS_VALUES_INPUT']

        self.first_dis_second_dis_neighbors_indices_input = data_dict['FIRST_DIS_SECOND_DIS_NEIGHBORS_INDICES_INPUT']
        self.first_dis_second_dis_neighbors_values_input_avg = data_dict['FIRST_DIS_SECOND_DIS_NEIGHBORS_VALUES_INPUT']

        self.low_att_layer_for_balance_neighbors_layer1 = {}
        self.low_att_layer_for_balance_neighbors_layer2 = {}

        self.first_trust_neighbors_values_input1 = {}
        self.first_distrust_neighbors_values_input1 = {}
        self.second_trust_neighbors_values_input2 = {}
        self.second_distrust_neighbors_values_input2 = {}

        self.first_trust_second_trust_neighbors_values_input = {}
        self.first_dis_second_dis_neighbors_values_input = {}

        for t in range(self.conf.total_len):# 0-10
            self.low_att_layer_for_balance_neighbors_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='low_att_BN_layer1_' + str(t))
            self.low_att_layer_for_balance_neighbors_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='low_att_BN_layer2_' + str(t))

        for t in range(self.conf.total_len):# 0-10
            # （++）
            self.first_trust_second_trust_neighbors_values_input[t] = tf.reduce_sum(tf.math.exp(self.low_att_layer_for_balance_neighbors_layer2[t](self.low_att_layer_for_balance_neighbors_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.first_trust_second_trust_neighbors_indices_input[t])], stddev=low_att_std)),[-1,1])      ) )  ),1)
            # （--）
            self.first_dis_second_dis_neighbors_values_input[t] = tf.reduce_sum(tf.math.exp(self.low_att_layer_for_balance_neighbors_layer2[t](self.low_att_layer_for_balance_neighbors_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.first_dis_second_dis_neighbors_indices_input[t])], stddev=low_att_std)),[-1,1])      ) )  ),1)
  


        #---------------- USER-ITEM ----------------#

        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']

        self.first_low_att_layer_for_user_item_layer1 = {}
        self.first_low_att_layer_for_user_item_layer2 = {}

        self.second_low_att_layer_for_user_item_layer1 = {}
        self.second_low_att_layer_for_user_item_layer2 = {}

        self.third_low_att_layer_for_user_item_layer1 = {}
        self.third_low_att_layer_for_user_item_layer2 = {}

        for t in range(self.conf.total_len):

            self.first_low_att_layer_for_user_item_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_UI_layer1_' + str(t))
            self.first_low_att_layer_for_user_item_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_UI_layer2_' + str(t))

            self.second_low_att_layer_for_user_item_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_UI_layer1_' + str(t))
            self.second_low_att_layer_for_user_item_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_UI_layer2_' + str(t))
            
            
            self.third_low_att_layer_for_user_item_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_UI_layer1_' + str(t))
            self.third_low_att_layer_for_user_item_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='third_low_att_UI_layer2_' + str(t))
            

        self.consumed_items_values_input1 = {}
        first_mean_consumed_item_influ_list = []
        first_var_consumed_item_influ_list = []

        self.consumed_items_values_input2 = {}
        self.consumed_items_values_input3 = {}

        for t in range(self.conf.total_len):

            # 1st_layer
            self.consumed_items_values_input1[t] = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_user_item_layer2[t](self.first_low_att_layer_for_user_item_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input[t])], stddev=low_att_std)),[-1,1])     \
                                                                                                                          ))   ),1)

            
            first_mean_consumed_item_influ, first_var_consumed_item_influ = tf.nn.moments(self.consumed_items_values_input1[t], axes=0)
            first_mean_consumed_item_influ_list.append(first_mean_consumed_item_influ)
            first_var_consumed_item_influ_list.append(first_var_consumed_item_influ)

            # 2nd_layer
            self.consumed_items_values_input2[t] = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_user_item_layer2[t](self.second_low_att_layer_for_user_item_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input[t])], stddev=low_att_std)),[-1,1])     \
                   
                                                                                                                   ))   ),1)
            
            # 3rd_layer
            self.consumed_items_values_input3[t] = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_user_item_layer2[t](self.third_low_att_layer_for_user_item_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input[t])], stddev=low_att_std)),[-1,1])     \
                                                                                                                        ))   ),1)
            
        self.first_user_item_low_att = [first_mean_consumed_item_influ_list, first_var_consumed_item_influ_list]


        #---------------- ITEM - USER ----------------#

        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']
        self.item_customer_values_input = data_dict['ITEM_CUSTOMER_VALUES_INPUT']

        self.first_low_att_layer_for_item_user_layer1 = {}
        self.first_low_att_layer_for_item_user_layer2 = {}

        self.second_low_att_layer_for_item_user_layer1 = {}
        self.second_low_att_layer_for_item_user_layer2 = {}

        self.third_low_att_layer_for_item_user_layer1 = {}
        self.third_low_att_layer_for_item_user_layer2 = {}

        for t in range(self.conf.total_len):
            self.first_low_att_layer_for_item_user_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_IU_layer1_'+str(t))
            self.first_low_att_layer_for_item_user_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_IU_layer2_'+str(t))

            self.second_low_att_layer_for_item_user_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_IU_layer1_'+str(t))
            self.second_low_att_layer_for_item_user_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_IU_layer2_'+str(t))

            
            self.third_low_att_layer_for_item_user_layer1[t] = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='third_low_att_IU_layer1_'+str(t))
            self.third_low_att_layer_for_item_user_layer2[t] = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='third_low_att_IU_layer2_'+str(t))
            

        self.item_customer_values_input1 = {}
        self.item_customer_values_input2 = {}
        self.item_customer_values_input3 = {}

        for t in range(self.conf.total_len):
            self.item_customer_values_input1[t] = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_item_user_layer2[t](self.first_low_att_layer_for_item_user_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input[t])], stddev=low_att_std)),[-1,1])     \
                                                                                                                            )  )    ),1)

            self.item_customer_values_input2[t] = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_item_user_layer2[t](self.second_low_att_layer_for_item_user_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input[t])], stddev=low_att_std)),[-1,1])     \
                                                                                                                            )  )    ),1)
            
            self.item_customer_values_input3[t] = tf.reduce_sum(tf.math.exp(self.third_low_att_layer_for_item_user_layer2[t](self.third_low_att_layer_for_item_user_layer1[t]( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input[t])], stddev=low_att_std)),[-1,1])     \
                                                                                                                            )  )    ),1)


        

        # prepare sparse matrix, in order to compute user's embedding from social neighbors and consumed items
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.item_customer_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)



        ######## Add Low Level Attention Here #########
        ###### First layer low att or avg ######

        ##### Avg_version #####
        
        #normal      
        self.social_neighbors_sparse_matrix_avg = {}
        self.consumed_items_sparse_matrix_avg = {}
        self.item_customer_sparse_matrix_avg = {}
       
        for t in range(self.conf.total_len):        
            self.social_neighbors_sparse_matrix_avg[t] = tf.SparseTensor(
                indices = self.social_neighbors_indices_input[t], 
                values = self.social_neighbors_values_input[t],
                dense_shape=self.social_neighbors_dense_shape
            ) 

            self.consumed_items_sparse_matrix_avg[t] = tf.SparseTensor(
                indices = self.consumed_items_indices_input[t], 
                values = self.consumed_items_values_input[t],
                dense_shape=self.consumed_items_dense_shape
            )

            self.item_customer_sparse_matrix_avg[t] = tf.SparseTensor(
                indices = self.item_customer_indices_input[t], 
                values = self.item_customer_values_input[t],
                dense_shape=self.item_customer_dense_shape
            )

        
        # Balance avg
        self.first_trust_second_trust_neighbors_sparse_matrix_avg = {}
        self.first_dis_second_dis_neighbors_sparse_matrix_avg = {}

        for t in range(self.conf.total_len):
            self.first_trust_second_trust_neighbors_sparse_matrix_avg[t] = tf.SparseTensor(
                indices = self.first_trust_second_trust_neighbors_indices_input[t], 
                values = self.first_trust_second_trust_neighbors_values_input_avg[t],
                dense_shape=self.social_neighbors_dense_shape
            )
            self.first_dis_second_dis_neighbors_sparse_matrix_avg[t] = tf.SparseTensor(
                indices = self.first_dis_second_dis_neighbors_indices_input[t], 
                values = self.first_dis_second_dis_neighbors_values_input_avg[t],
                dense_shape=self.social_neighbors_dense_shape
            )
        

        ##### Low att_version #####
        
        #normal
        self.first_layer_social_neighbors_sparse_matrix = {}
        self.first_layer_consumed_items_sparse_matrix = {}
        self.first_layer_item_customer_sparse_matrix = {}

        self.first_social_neighbors_low_level_att_matrix = {}
        self.first_consumed_items_low_level_att_matrix = {}
        self.first_items_users_neighborslow_level_att_matrix = {}

        for t in range(self.conf.total_len):
            self.first_layer_social_neighbors_sparse_matrix[t] = tf.SparseTensor(
                indices = self.social_neighbors_indices_input[t], 
                values = self.social_neighbors_values_input1[t],
                dense_shape=self.social_neighbors_dense_shape
            )
            
            self.first_layer_consumed_items_sparse_matrix[t] = tf.SparseTensor(
                indices = self.consumed_items_indices_input[t], 
                values = self.consumed_items_values_input1[t],
                dense_shape=self.consumed_items_dense_shape
            )
            
            self.first_layer_item_customer_sparse_matrix[t] = tf.SparseTensor(
                indices = self.item_customer_indices_input[t], 
                values = self.item_customer_values_input1[t],
                dense_shape=self.item_customer_dense_shape
            )
        
            self.first_social_neighbors_low_level_att_matrix[t] = tf.sparse.softmax(self.first_layer_social_neighbors_sparse_matrix[t]) 
            self.first_consumed_items_low_level_att_matrix[t] = tf.sparse.softmax(self.first_layer_consumed_items_sparse_matrix[t]) 
            self.first_items_users_neighborslow_level_att_matrix[t] = tf.sparse.softmax(self.first_layer_item_customer_sparse_matrix[t]) 
        

        #Balance low attention
        self.first_trust_second_trust_neighbors_sparse_matrix = {}
        self.first_dis_second_dis_neighbors_sparse_matrix = {}

        self.first_trust_second_trust_neighbors_low_level_att_matrix = {}
        self.first_dis_second_dis_neighbors_low_level_att_matrix = {}

        for t in range(self.conf.total_len):
            self.first_trust_second_trust_neighbors_sparse_matrix[t] = tf.SparseTensor(
                indices = self.first_trust_second_trust_neighbors_indices_input[t], 
                values = self.first_trust_second_trust_neighbors_values_input[t],
                dense_shape=self.social_neighbors_dense_shape
            )
            
            self.first_dis_second_dis_neighbors_sparse_matrix[t] = tf.SparseTensor(
                indices = self.first_dis_second_dis_neighbors_indices_input[t], 
                values = self.first_dis_second_dis_neighbors_values_input[t],
                dense_shape=self.social_neighbors_dense_shape
            )

            self.first_trust_second_trust_neighbors_low_level_att_matrix[t] = tf.sparse.softmax(self.first_trust_second_trust_neighbors_sparse_matrix[t]) 
            self.first_dis_second_dis_neighbors_low_level_att_matrix[t] = tf.sparse.softmax(self.first_dis_second_dis_neighbors_sparse_matrix[t]) 


        
        ###### Second layer ######
        #normal
        self.second_layer_social_neighbors_sparse_matrix = {}
        self.second_layer_consumed_items_sparse_matrix = {}
        self.second_layer_item_customer_sparse_matrix = {}

        self.second_social_neighbors_low_level_att_matrix = {}
        self.second_consumed_items_low_level_att_matrix = {}
        self.second_items_users_neighborslow_level_att_matrix = {}

        # low att
        for t in range(self.conf.total_len):
        
            self.second_layer_social_neighbors_sparse_matrix[t] = tf.SparseTensor(
                indices = self.social_neighbors_indices_input[t], 
                values = self.social_neighbors_values_input2[t],
                dense_shape=self.social_neighbors_dense_shape
            )
            self.second_layer_consumed_items_sparse_matrix[t] = tf.SparseTensor(
                indices = self.consumed_items_indices_input[t], 
                values = self.consumed_items_values_input2[t],
                dense_shape=self.consumed_items_dense_shape
            )
            self.second_layer_item_customer_sparse_matrix[t] = tf.SparseTensor(
                indices = self.item_customer_indices_input[t], 
                values = self.item_customer_values_input2[t],
                dense_shape=self.item_customer_dense_shape
            )

            self.second_social_neighbors_low_level_att_matrix[t] = tf.sparse.softmax(self.second_layer_social_neighbors_sparse_matrix[t]) 
            self.second_consumed_items_low_level_att_matrix[t] = tf.sparse.softmax(self.second_layer_consumed_items_sparse_matrix[t]) 
            self.second_items_users_neighborslow_level_att_matrix[t] = tf.sparse.softmax(self.second_layer_item_customer_sparse_matrix[t]) 
  
        
        
        
        ###### Third layer low att ######
 
        self.third_layer_social_neighbors_sparse_matrix = {}
        self.third_layer_consumed_items_sparse_matrix = {}
        self.third_layer_item_customer_sparse_matrix = {}

        self.third_social_neighbors_low_level_att_matrix = {}
        self.third_consumed_items_low_level_att_matrix = {}
        self.third_items_users_neighborslow_level_att_matrix = {}

        # low att
        for t in range(self.conf.total_len):
        
            self.third_layer_social_neighbors_sparse_matrix[t] = tf.SparseTensor(
                indices = self.social_neighbors_indices_input[t], 
                values = self.social_neighbors_values_input3[t],
                dense_shape=self.social_neighbors_dense_shape
            )
            self.third_layer_consumed_items_sparse_matrix[t] = tf.SparseTensor(
                indices = self.consumed_items_indices_input[t], 
                values = self.consumed_items_values_input3[t],
                dense_shape=self.consumed_items_dense_shape
            )
            self.third_layer_item_customer_sparse_matrix[t] = tf.SparseTensor(
                indices = self.item_customer_indices_input[t], 
                values = self.item_customer_values_input3[t],
                dense_shape=self.item_customer_dense_shape
            )
            
            self.third_social_neighbors_low_level_att_matrix[t] = tf.sparse.softmax(self.third_layer_social_neighbors_sparse_matrix[t]) 
            self.third_consumed_items_low_level_att_matrix[t] = tf.sparse.softmax(self.third_layer_consumed_items_sparse_matrix[t]) 
            self.third_items_users_neighborslow_level_att_matrix[t] = tf.sparse.softmax(self.third_layer_item_customer_sparse_matrix[t]) 
        
        

    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y


    #---------------- AVG ----------------#

    
    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = []
        # avg
        
        for t in range(self.conf.total_len):
            user_embedding_from_social_neighbors.append(tf.sparse_tensor_dense_matmul(self.social_neighbors_sparse_matrix_avg[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_social_neighbors,[self.conf.total_len, self.conf.num_users, self.conf.dimension])
        

    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = []

        #avg
        
        for t in range(self.conf.total_len):
            user_embedding_from_consumed_items.append(tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix_avg[t], current_item_embedding[t]))

        return tf.reshape(user_embedding_from_consumed_items,[self.conf.total_len,self.conf.num_users,self.conf.dimension])
        

    def generateItemEmebddingFromCustomer(self, current_user_embedding):
        item_embedding_from_customer = []
        #avg
        
        for t in range(self.conf.total_len):
            item_embedding_from_customer.append(tf.sparse_tensor_dense_matmul(self.item_customer_sparse_matrix_avg[t], current_user_embedding[t]))

        return tf.reshape(item_embedding_from_customer, [self.conf.total_len,self.conf.num_items,self.conf.dimension])    
        
        
    ##### Blance #####
    # avg  
    def generateUserEmebddingFromFirsttrustSecondtrustNeighbors(self, current_user_embedding):# （++）set
        user_embedding_from_first_trust_second_trust_neighbors = []
        #avg
        for t in range(self.conf.total_len):
            user_embedding_from_first_trust_second_trust_neighbors.append(tf.sparse_tensor_dense_matmul(self.first_trust_second_trust_neighbors_sparse_matrix_avg[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_first_trust_second_trust_neighbors, [self.conf.total_len, self.conf.num_users, self.conf.dimension]) 

    def generateUserEmebddingFromFirstdisSeconddisNeighbors(self, current_user_embedding):# （--）set
        user_embedding_from_first_dis_second_dis_neighbors = []
        #avg
        for t in range(self.conf.total_len):
            user_embedding_from_first_dis_second_dis_neighbors.append(tf.sparse_tensor_dense_matmul(self.first_dis_second_dis_neighbors_sparse_matrix_avg[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_first_dis_second_dis_neighbors, [self.conf.total_len, self.conf.num_users, self.conf.dimension])  


    
    #---------------- low - attention ----------------#

    def generateUserEmbeddingFromSocialNeighbors1(self, current_user_embedding):
        user_embedding_from_social_neighbors1 = []

        for t in range(self.conf.total_len):
            user_embedding_from_social_neighbors1.append(tf.sparse_tensor_dense_matmul(self.first_social_neighbors_low_level_att_matrix[t], current_user_embedding[t]))     

        return tf.reshape(user_embedding_from_social_neighbors1,[self.conf.total_len, self.conf.num_users, self.conf.dimension])

    
    
    def generateUserEmebddingFromConsumedItems1(self, current_item_embedding):
        user_embedding_from_consumed_items1 = []

        for t in range(self.conf.total_len):
            user_embedding_from_consumed_items1.append(tf.sparse_tensor_dense_matmul(self.first_consumed_items_low_level_att_matrix[t], current_item_embedding[t]))

        return tf.reshape(user_embedding_from_consumed_items1,[self.conf.total_len,self.conf.num_users,self.conf.dimension])
    

    def generateItemEmebddingFromCustomer1(self, current_user_embedding):
        item_embedding_from_customer1 = []

        for t in range(self.conf.total_len):
            item_embedding_from_customer1.append(tf.sparse_tensor_dense_matmul(self.first_items_users_neighborslow_level_att_matrix[t], current_user_embedding[t]))

        return tf.reshape(item_embedding_from_customer1, [self.conf.total_len,self.conf.num_items,self.conf.dimension])    
    

    # low ATT
    
    def generateUserEmbeddingFromSocialNeighbors2(self, current_user_embedding):
        user_embedding_from_social_neighbors2 = []

        for t in range(self.conf.total_len):
            user_embedding_from_social_neighbors2.append(tf.sparse_tensor_dense_matmul(self.second_social_neighbors_low_level_att_matrix[t], current_user_embedding[t]))     

        return tf.reshape(user_embedding_from_social_neighbors2,[self.conf.total_len, self.conf.num_users, self.conf.dimension])
    
    
    def generateUserEmebddingFromConsumedItems2(self, current_item_embedding):
        user_embedding_from_consumed_items2 = []

        for t in range(self.conf.total_len):
            user_embedding_from_consumed_items2.append(tf.sparse_tensor_dense_matmul(self.second_consumed_items_low_level_att_matrix[t], current_item_embedding[t]))

        return tf.reshape(user_embedding_from_consumed_items2,[self.conf.total_len,self.conf.num_users,self.conf.dimension])

    def generateItemEmebddingFromCustomer2(self, current_user_embedding):
        item_embedding_from_customer2 = []

        for t in range(self.conf.total_len):
            item_embedding_from_customer2.append(tf.sparse_tensor_dense_matmul(self.second_items_users_neighborslow_level_att_matrix[t], current_user_embedding[t]))

        return tf.reshape(item_embedding_from_customer2, [self.conf.total_len,self.conf.num_items,self.conf.dimension])   


    # low att
    
    def generateUserEmbeddingFromSocialNeighbors3(self, current_user_embedding):
        user_embedding_from_social_neighbors3 = []

        for t in range(self.conf.total_len):
            user_embedding_from_social_neighbors3.append(tf.sparse_tensor_dense_matmul(self.third_social_neighbors_low_level_att_matrix[t], current_user_embedding[t]))     

        return tf.reshape(user_embedding_from_social_neighbors3,[self.conf.total_len, self.conf.num_users, self.conf.dimension])
    
    def generateUserEmebddingFromConsumedItems3(self, current_item_embedding):
        user_embedding_from_consumed_items3 = []

        for t in range(self.conf.total_len):
            user_embedding_from_consumed_items3.append(tf.sparse_tensor_dense_matmul(self.third_consumed_items_low_level_att_matrix[t], current_item_embedding[t]))

        return tf.reshape(user_embedding_from_consumed_items3,[self.conf.total_len,self.conf.num_users,self.conf.dimension])

    def generateItemEmebddingFromCustomer3(self, current_user_embedding):
        item_embedding_from_customer3 = []

        for t in range(self.conf.total_len):
            item_embedding_from_customer3.append(tf.sparse_tensor_dense_matmul(self.third_items_users_neighborslow_level_att_matrix[t], current_user_embedding[t]))

        return tf.reshape(item_embedding_from_customer3, [self.conf.total_len,self.conf.num_items,self.conf.dimension])   
    
    
    ##### Blance #####
    # low-att
    def generateUserEmebddingFromFirsttrustSecondtrustNeighbors1(self, current_user_embedding):# （++）set
        user_embedding_from_first_trust_second_trust_neighbors = []

        for t in range(self.conf.total_len):
            user_embedding_from_first_trust_second_trust_neighbors.append(tf.sparse_tensor_dense_matmul(self.first_trust_second_trust_neighbors_low_level_att_matrix[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_first_trust_second_trust_neighbors, [self.conf.total_len, self.conf.num_users, self.conf.dimension]) 
    
    def generateUserEmebddingFromFirsttrustSecondtrustNeighbors2(self, current_user_embedding):# （++）set
        user_embedding_from_first_trust_second_trust_neighbors = []

        for t in range(self.conf.total_len):
            user_embedding_from_first_trust_second_trust_neighbors.append(tf.sparse_tensor_dense_matmul(self.first_trust_second_trust_neighbors_low_level_att_matrix[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_first_trust_second_trust_neighbors, [self.conf.total_len, self.conf.num_users, self.conf.dimension]) 

    def generateUserEmebddingFromFirstdisSeconddisNeighbors1(self, current_user_embedding):# （--）set
        user_embedding_from_first_dis_second_dis_neighbors = []
        
        for t in range(self.conf.total_len):
            user_embedding_from_first_dis_second_dis_neighbors.append(tf.sparse_tensor_dense_matmul(self.first_dis_second_dis_neighbors_low_level_att_matrix[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_first_dis_second_dis_neighbors, [self.conf.total_len, self.conf.num_users, self.conf.dimension])  
    
    def generateUserEmebddingFromFirstdisSeconddisNeighbors2(self, current_user_embedding):# （--）set
        user_embedding_from_first_dis_second_dis_neighbors = []
        
        for t in range(self.conf.total_len):
            user_embedding_from_first_dis_second_dis_neighbors.append(tf.sparse_tensor_dense_matmul(self.first_dis_second_dis_neighbors_low_level_att_matrix[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_first_dis_second_dis_neighbors, [self.conf.total_len, self.conf.num_users, self.conf.dimension])  

    
    
    def initializeNodes(self):
        # Rating
        self.item_input = tf.placeholder("int32", [None, 1]) # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1]) # Get user embedding from the core_user_input
        self.labels_input = tf.placeholder("float32", [None, 1])
        self.r_stamp_input = tf.placeholder("int32", [None, 1])

        self.eva_item_input = tf.placeholder("int32", [None, 1]) # Get item embedding from the core_item_input
        self.eva_user_input = tf.placeholder("int32", [None, 1]) # Get user embedding from the core_user_input
        #self.eva_labels_input = tf.placeholder("float32", [None, 1])


        # Social
        self.user1_input = tf.placeholder("int32", [None, 1]) 
        self.user2_input = tf.placeholder("int32", [None, 1]) 
        self.s_labels_input = tf.placeholder("float32", [None, 1])
        self.s_stamp_input = tf.placeholder("int32", [None, 1])

        self.eva_user1_input = tf.placeholder("int32", [None, 1]) 
        self.eva_user2_input = tf.placeholder("int32", [None, 1]) 
        #self.eva_s_labels_input = tf.placeholder("float32", [None, 1])

        stddev_std = 0.01
        # For Temporal information
        # supply for T=0-10
        self.user_social_temporal_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.total_len, self.conf.dimension], stddev=stddev_std), name='user_social_embedding')
        self.user_temporal_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.total_len, self.conf.dimension], stddev=stddev_std), name='user_embedding')
        self.item_temporal_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.total_len, self.conf.dimension], stddev=stddev_std), name='item_embedding')
        
        self.user_social_ini_embedding = tf.Variable(tf.random_normal([self.conf.num_users, 1, self.conf.dimension], stddev=stddev_std), name='user_social_ini_embedding')
        self.user_ini_embedding = tf.Variable(tf.random_normal([self.conf.num_users, 1, self.conf.dimension], stddev=stddev_std), name='user_ini_embedding')
        self.item_ini_embedding = tf.Variable(tf.random_normal([self.conf.num_items, 1, self.conf.dimension], stddev=stddev_std), name='item_ini_embedding')

        #每个（conf.total_len+1）时序都有相同的初始化embedding - static
        self.user_social_ini_embedding = tf.tile(self.user_social_ini_embedding, (1,self.conf.total_len+1,1))
        self.user_ini_embedding = tf.tile(self.user_ini_embedding, (1,self.conf.total_len+1,1))
        self.item_ini_embedding = tf.tile(self.item_ini_embedding, (1,self.conf.total_len+1,1))

        # supply for T=11 
        self.user_social_temporal_embedding_T11 = tf.Variable(
            tf.random_normal([self.conf.num_users, 1, self.conf.dimension], stddev=stddev_std), name='user_social_embedding_T11')
        self.user_temporal_embedding_T11 = tf.Variable(
            tf.random_normal([self.conf.num_users, 1, self.conf.dimension], stddev=stddev_std), name='user_embedding_T11')
        self.item_temporal_embedding_T11 = tf.Variable(
            tf.random_normal([self.conf.num_items, 1, self.conf.dimension], stddev=stddev_std), name='item_embedding_T11')

        # make T the first dimension for fusion

        self.user_social_temporal_embedding_T11 = tf.transpose(self.user_social_temporal_embedding_T11, [1, 0, 2])   # t u d
        self.user_temporal_embedding_T11 = tf.transpose(self.user_temporal_embedding_T11, [1, 0, 2])   # t u d
        self.item_temporal_embedding_T11 = tf.transpose(self.item_temporal_embedding_T11, [1, 0, 2])   # t u d

        #set_trace()

        # This part for the tranditional MF
        self.shallow_user_embedding = tf.Variable(tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=stddev_std), name='shallow_user_embedding')
        self.shallow_item_embedding = tf.Variable(tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=stddev_std), name='shallow_item_embedding')
        self.shallow_social_user_embedding = tf.Variable(tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=stddev_std), name='shallow_social_user_embedding')


        self.reduce_dimension_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='reduce_dimension_layer')
        
        self.balance_user_graph_att_layer1_1 = {}
        self.balance_user_graph_att_layer1_2 = {}
        self.balance_user_graph_att_layer2_1 = {}
        self.balance_user_graph_att_layer2_2 = {}

        for t in range(self.conf.total_len):
            self.balance_user_graph_att_layer1_1[t] = tf.layers.Dense(\
                1, activation=tf.nn.tanh, name='firstGCN_balance_MLP_first_layer')
            self.balance_user_graph_att_layer1_2[t] = tf.layers.Dense(\
                1, activation=tf.nn.leaky_relu, name='firstGCN_balance_MLP_second_layer')

            self.balance_user_graph_att_layer2_1[t] = tf.layers.Dense(\
                1, activation=tf.nn.tanh, name='secondGCN_balance_MLP_first_layer')
            self.balance_user_graph_att_layer2_2[t] = tf.layers.Dense(\
                1, activation=tf.nn.leaky_relu, name='secondGCN_balance_MLP_second_layer')

        

        self.social_first_MLP = tf.layers.Dense(self.conf.dimension*4, activation=tf.nn.tanh, name='s_first_mlp')
        self.social_second_MLP = tf.layers.Dense(self.conf.dimension*2, activation=tf.nn.tanh, name='s_second_mlp')
        self.social_third_MLP = tf.layers.Dense(self.conf.dimension, activation=tf.nn.tanh, name='s_third_mlp')
        self.y_s = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='s_output')

        self.rec_first_MLP = tf.layers.Dense(self.conf.dimension*4, activation=tf.nn.tanh, name='r_first_mlp')
        self.rec_second_MLP = tf.layers.Dense(self.conf.dimension*2, activation=tf.nn.tanh, name='r_second_mlp')
        self.rec_third_MLP = tf.layers.Dense(self.conf.dimension, activation=tf.nn.tanh, name='r_third_mlp')
        self.y_r = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='r_output')

    
    def constructTrainGraph(self):

        # Temporal embedding should put T as the first dimension
        self.temporal_user_social_embedding = tf.transpose(self.user_social_temporal_embedding, [1, 0, 2])   # t u d    original u t d

        self.temporal_user_embedding = tf.transpose(self.user_temporal_embedding, [1, 0, 2])   # t u d   cuz it has been written double time, need to be checked
        self.temporal_item_embedding = tf.transpose(self.item_temporal_embedding, [1, 0, 2])   # t v d

        self.temporal_user_embedding = tf.transpose(self.user_temporal_embedding, [1, 0, 2])   # t u d
        self.temporal_item_embedding = tf.transpose(self.item_temporal_embedding, [1, 0, 2])   # t v d （11，26991，10）

        # First Layer
        # normal
        T_first_user_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors1(self.temporal_user_social_embedding)
        T_first_user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems1(self.temporal_item_embedding)
        T_first_item_embedding_from_customer = self.generateItemEmebddingFromCustomer1(self.temporal_user_embedding)

        #self.first_gcn_user_social_embedding  =  T_first_user_embedding_from_social_neighbors
        self.first_gcn_user_embedding_from_consumed_items = T_first_user_embedding_from_consumed_items
        self.first_gcn_item_embedding_from_customer = T_first_item_embedding_from_customer
        
        # first layer node-attention 
        self.T_first_social_user_embedding_from_first_trust_second_trust_neighbors = self.generateUserEmebddingFromFirsttrustSecondtrustNeighbors1(self.temporal_user_social_embedding)
        self.T_first_social_user_embedding_from_first_dis_second_dis_neighbors = self.generateUserEmebddingFromFirstdisSeconddisNeighbors1(self.temporal_user_social_embedding) 
 
        # first layer graph-attention
        self.social_neighbors_attention_1 = {}
        self.balance_neighbors_attention_1 = {}
        first_gcn_user_social_embedding = []
        for t in range(self.conf.total_len):

            social_neighbors_attention = tf.math.exp(self.balance_user_graph_att_layer1_2[t](self.balance_user_graph_att_layer1_1[t](\
                                    tf.concat([T_first_user_embedding_from_social_neighbors[t], self.temporal_user_social_embedding[t]], -1))))

            balance_neighbors_attention = tf.math.exp(self.balance_user_graph_att_layer1_2[t](self.balance_user_graph_att_layer1_1[t](\
                                    tf.concat([self.T_first_social_user_embedding_from_first_dis_second_dis_neighbors[t], self.temporal_user_social_embedding[t]], -1))))

            sum_attention = social_neighbors_attention + balance_neighbors_attention
            self.social_neighbors_attention_1[t] = social_neighbors_attention / sum_attention
            self.balance_neighbors_attention_1[t] = balance_neighbors_attention / sum_attention

            first_gcn_user_social_embedding.append((self.social_neighbors_attention_1[t] * T_first_user_embedding_from_social_neighbors[t]\
                                                + self.balance_neighbors_attention_1[t] * self.T_first_social_user_embedding_from_first_dis_second_dis_neighbors[t])\
                                                + 0.5*self.temporal_user_social_embedding[t])     
        
        self.first_gcn_user_social_embedding = tf.reshape(first_gcn_user_social_embedding, [self.conf.total_len,self.conf.num_users,self.conf.dimension])   
        
        # first layer graph-avg
        # self.first_gcn_user_social_embedding  = T_first_user_embedding_from_social_neighbors   + self.T_first_social_user_embedding_from_first_dis_second_dis_neighbors+self.T_first_social_user_embedding_from_first_trust_second_trust_neighbors + self.temporal_user_social_embedding

        
        # Second Layer
        # normal
        T_second_user_social_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors2(self.first_gcn_user_social_embedding)
        T_second_user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems2(self.first_gcn_item_embedding_from_customer)
        T_second_item_embedding_from_customer = self.generateItemEmebddingFromCustomer2(self.first_gcn_user_embedding_from_consumed_items)

        #self.second_gcn_user_social_embedding =  T_second_user_social_embedding_from_social_neighbors
        self.second_gcn_user_embedding_from_consumed_items = T_second_user_embedding_from_consumed_items
        self.second_gcn_item_embedding_from_customer =  T_second_item_embedding_from_customer

        # second layer node-attention 
        self.T_second_user_embedding_from_first_dis_second_dis_neighbors = self.generateUserEmebddingFromFirstdisSeconddisNeighbors2(self.first_gcn_user_social_embedding) 
        self.T_second_user_embedding_from_first_trust_second_trust_neighbors = self.generateUserEmebddingFromFirsttrustSecondtrustNeighbors2(self.first_gcn_user_social_embedding) 
        
        # second layer graph-attention
        self.social_neighbors_attention_2 = {}
        self.balance_neighbors_attention_2 = {}
        second_gcn_user_social_embedding = []
        for t in range(self.conf.total_len):
            social_neighbors_attention = tf.math.exp(self.balance_user_graph_att_layer2_2[t](self.balance_user_graph_att_layer2_1[t](\
                                    tf.concat([T_second_user_social_embedding_from_social_neighbors[t], self.first_gcn_user_social_embedding[t]], 1))))

            balance_neighbors_attention = tf.math.exp(self.balance_user_graph_att_layer2_2[t](self.balance_user_graph_att_layer2_1[t](\
                                    tf.concat([self.T_second_user_embedding_from_first_dis_second_dis_neighbors[t], self.first_gcn_user_social_embedding[t]], 1))))

            sum_attention = social_neighbors_attention + balance_neighbors_attention
            self.social_neighbors_attention_2[t] = social_neighbors_attention / sum_attention
            self.balance_neighbors_attention_2[t] = balance_neighbors_attention / sum_attention

            second_gcn_user_social_embedding.append(0.5*(self.social_neighbors_attention_2[t] * T_second_user_social_embedding_from_social_neighbors[t]\
                                                    + self.balance_neighbors_attention_2[t] * self.T_second_user_embedding_from_first_dis_second_dis_neighbors[t])\
                                                    + 0.5*self.first_gcn_user_social_embedding[t])

        self.second_gcn_user_social_embedding = tf.reshape(second_gcn_user_social_embedding, [self.conf.total_len,self.conf.num_users,self.conf.dimension])

        # second layer graph-avg
        # self.second_gcn_user_social_embedding =  T_second_user_social_embedding_from_social_neighbors   +  self.T_second_user_embedding_from_first_dis_second_dis_neighbors + self.T_second_user_embedding_from_first_trust_second_trust_neighbors + self.first_gcn_user_social_embedding


        # third Layer
        T_third_user_social_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors3(self.second_gcn_user_social_embedding)
        T_third_user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems3(self.second_gcn_item_embedding_from_customer)
        T_third_item_embedding_from_customer = self.generateItemEmebddingFromCustomer3(self.second_gcn_user_embedding_from_consumed_items)

        self.third_gcn_user_social_embedding =  self.second_gcn_user_social_embedding + T_third_user_social_embedding_from_social_neighbors
        self.third_gcn_user_embedding_from_consumed_items = self.second_gcn_user_embedding_from_consumed_items + T_third_user_embedding_from_consumed_items
        self.third_gcn_item_embedding_from_customer = self.second_gcn_item_embedding_from_customer +  T_third_item_embedding_from_customer
        

        # T0-10 : last embedding
        user_social_last_embedding_T010 = self.second_gcn_user_social_embedding  + self.first_gcn_user_social_embedding + self.temporal_user_social_embedding #+ self.third_gcn_user_social_embedding
        
        user_last_embedding_T010 = self.first_gcn_user_embedding_from_consumed_items + self.temporal_user_embedding + self.second_gcn_user_embedding_from_consumed_items #+ self.third_gcn_user_embedding_from_consumed_items
                              
        item_last_embedding_T010 = self.first_gcn_item_embedding_from_customer + self.temporal_item_embedding + self.second_gcn_item_embedding_from_customer #+ self.third_gcn_item_embedding_from_customer
                              

        # fuse T11 for RNN
        user_social_last_embedding = tf.concat([user_social_last_embedding_T010, self.user_social_temporal_embedding_T11], 0)
        user_last_embedding = tf.concat([user_last_embedding_T010, self.user_temporal_embedding_T11], 0)
        item_last_embedding = tf.concat([item_last_embedding_T010, self.item_temporal_embedding_T11], 0)

        
        # GRU for embedding
        userSocialCell = tf.nn.rnn_cell.GRUCell(num_units=self.conf.dimension, name='userSocialCell')
        #userSocialCell = rnn_cell.LayerNormBasicLSTMCell(num_units=self.conf.dimension)
        #userInput = user_last_embedding   # [t_step, user_num, dim]
        userSocialInput = tf.transpose(user_social_last_embedding, [1, 0, 2])   # because dynamic_rnn needs time_len as the second input; u-t-d
        userSocialOutputs, userSocialStates = tf.nn.dynamic_rnn(cell=userSocialCell, inputs=userSocialInput, dtype=tf.float32, time_major=False)
        self.userSocialOutputs = userSocialOutputs  # output is a multi dimension array
        self.userSocialStates = userSocialStates    # the last state of the output

        userCell = tf.nn.rnn_cell.GRUCell(num_units=self.conf.dimension, name='userCell')
        #userCell = rnn_cell.LayerNormBasicLSTMCell(num_units=self.conf.dimension)
        #userInput = user_last_embedding   # [t_step, user_num, dim]
        userInput = tf.transpose(user_last_embedding, [1, 0, 2])
        userOutputs, userStates = tf.nn.dynamic_rnn(userCell, userInput, dtype=tf.float32, time_major=False)
        self.userOutputs = userOutputs
        self.userStates = userStates

        itemCell = tf.nn.rnn_cell.GRUCell(num_units=self.conf.dimension, name='itemCell')
        #itemCell = rnn_cell.LayerNormBasicLSTMCell(num_units=self.conf.dimension)
        #itemInput = item_last_embedding   # [t_step, item_num, dim]
        itemInput = tf.transpose(item_last_embedding, [1, 0, 2])
        itemOutputs, itemStates = tf.nn.dynamic_rnn(itemCell, itemInput, dtype=tf.float32, time_major=False)
        self.itemOutputs = itemOutputs
        self.itemStates = itemStates

        
        # final embedding
        self.final_social_user_embedding = self.user_social_ini_embedding+self.userSocialOutputs[:,0:self.conf.total_len+1,:]
        #self.final_social_user_embedding = self.userSocialOutputs#[:,0:self.conf.total_len+1,:]
        # [user_num, 11+1, dim]
        #self.final_user_embedding = tf.concat([self.user_ini_embedding, self.userOutputs], 1)
        self.final_user_embedding = self.user_ini_embedding+self.userOutputs[:,0:self.conf.total_len+1,:]
        #self.final_user_embedding = self.userOutputs#[:,self.conf.total_len+1,:]
        # [item_num, 11+1, dim]
        self.final_item_embedding = self.item_ini_embedding+self.itemOutputs[:,0:self.conf.total_len+1,:]
        #self.final_item_embedding = self.itemOutputs#[:,0:self.conf.total_len+1,:]
        
        
        
        # --- Embedding Extracting --- #
        # social part
        # basic here means the preference embeddding
        user1_social_stamp_indices=tf.concat([tf.reshape(self.user1_input,[-1,1]),tf.reshape(self.s_stamp_input,[-1,1])],1)
        user2_social_stamp_indices=tf.concat([tf.reshape(self.user2_input,[-1,1]),tf.reshape(self.s_stamp_input,[-1,1])],1)
        
        deep_social_user1_latent = tf.gather_nd(self.final_social_user_embedding, user1_social_stamp_indices) #S
        deep_social_user2_latent = tf.gather_nd(self.final_social_user_embedding, user2_social_stamp_indices)

        mf_social_user1_latent = tf.gather_nd(self.shallow_social_user_embedding, self.user1_input)  #X
        mf_social_user2_latent = tf.gather_nd(self.shallow_social_user_embedding, self.user2_input)


        deep_basic_user1_latent = tf.gather_nd(self.final_user_embedding, user1_social_stamp_indices) #P
        deep_basic_user2_latent = tf.gather_nd(self.final_user_embedding, user2_social_stamp_indices)

        user1_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.user1_input) #W 
        user2_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.user2_input)

        # rating part
        user_stamp_indices=tf.concat([tf.reshape(self.user_input,[-1,1]),tf.reshape(self.r_stamp_input,[-1,1])],1)
        item_stamp_indices=tf.concat([tf.reshape(self.item_input,[-1,1]),tf.reshape(self.r_stamp_input,[-1,1])],1)

        deep_user_latent = tf.gather_nd(self.final_user_embedding, user_stamp_indices)    # P
        deep_social_latent = tf.gather_nd(self.final_social_user_embedding, user_stamp_indices)   #S
        deep_item_latent = tf.gather_nd(self.final_item_embedding, item_stamp_indices)  # Q

        mf_user_latent = tf.gather_nd(self.shallow_user_embedding, self.user_input)  # W
        mf_social_latent = tf.gather_nd(self.shallow_social_user_embedding, self.user_input) #X
        mf_item_latent = tf.gather_nd(self.shallow_item_embedding, self.item_input) # M


        #--- Training model ---#

        # --- Feature combination, fusion Layer --- #
        # Left user1-user2 part
        #(deep)
        user1_user2_basic_concat = tf.concat([deep_basic_user1_latent, deep_basic_user2_latent],1)
        user1_user2_social_concat = tf.concat([deep_social_user1_latent, deep_social_user2_latent], 1)
        total_user1_user2_left_concat = tf.concat([user1_user2_social_concat, user1_user2_basic_concat],1)
        #(shallow)
        user1_user2_basic_multiply =tf.multiply(user1_basic_mf_latent, user2_basic_mf_latent)
        user1_user2_social_multiply = tf.multiply(mf_social_user1_latent, mf_social_user2_latent) 
        total_user1_user2_social_basic_multiply_concat_left = tf.concat([user1_user2_social_multiply, user1_user2_basic_multiply],1)


        # Right user-item part
        #(deep)
        user_social_item_concat = tf.concat([deep_social_latent, deep_item_latent],1)
        user_basic_item_concat = tf.concat([deep_user_latent, deep_item_latent],1)
        total_user_item_concat = tf.concat([user_social_item_concat, user_basic_item_concat], 1)
        #(shallow)
        user_social_item_multiply = tf.multiply(mf_social_latent, mf_item_latent)
        user_basic_item_multiply = tf.multiply(mf_user_latent,mf_item_latent)
        user_item_multiply = tf.concat([user_social_item_multiply,user_basic_item_multiply],1)


        # --- MLP LAYER for Prediction --- #
    
        #social prediction
        if self.conf.layer_depth == 1:
            s_mlp_output = self.social_first_MLP(total_user1_user2_left_concat)

        if self.conf.layer_depth == 2:
            s_mlp_output = self.social_second_MLP(self.social_first_MLP(total_user1_user2_left_concat))

        if self.conf.layer_depth == 3:
            s_mlp_output = self.social_third_MLP(self.social_second_MLP(self.social_first_MLP(total_user1_user2_left_concat)))


        
        self.s_prediction = self.y_s(tf.concat([s_mlp_output,total_user1_user2_social_basic_multiply_concat_left],1))

        #rating prediction
        if self.conf.layer_depth == 1:
            r_mlp_output = self.rec_first_MLP(total_user_item_concat)

        if self.conf.layer_depth == 2:
            r_mlp_output = self.rec_second_MLP(self.rec_first_MLP(total_user_item_concat))

        if self.conf.layer_depth == 3:
            r_mlp_output = self.rec_third_MLP(self.rec_second_MLP(self.rec_first_MLP(total_user_item_concat)))

        self.prediction = self.y_r(tf.concat([r_mlp_output,user_item_multiply],1))        

     
     


        # --- EVA PART --- #

        #eva_deep_social_user1_embedding = tf.gather_nd(self.userSocialStates, self.user1_input)
        #eva_deep_social_user2_embedding = tf.gather_nd(self.userSocialStates, self.user2_input)

       
        # social part
        eva_deep_social_user1_latent = tf.gather_nd(self.final_social_user_embedding[:,self.conf.total_len,:], self.eva_user1_input) #use the last stamp for testing
        eva_deep_social_user2_latent = tf.gather_nd(self.final_social_user_embedding[:,self.conf.total_len,:], self.eva_user2_input)

        eva_mf_social_user1_latent = tf.gather_nd(self.shallow_social_user_embedding, self.eva_user1_input)
        eva_mf_social_user2_latent = tf.gather_nd(self.shallow_social_user_embedding, self.eva_user2_input)


        eva_deep_basic_user1_latent = tf.gather_nd(self.final_user_embedding[:,self.conf.total_len,:], self.eva_user1_input)
        eva_deep_basic_user2_latent = tf.gather_nd(self.final_user_embedding[:,self.conf.total_len,:], self.eva_user2_input)

        eva_user1_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.eva_user1_input)
        eva_user2_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.eva_user2_input)

        # rating part
        eva_deep_user_latent = tf.gather_nd(self.final_user_embedding[:,self.conf.total_len,:], self.eva_user_input)  
        eva_deep_social_latent = tf.gather_nd(self.final_social_user_embedding[:,self.conf.total_len,:], self.eva_user_input)  
        eva_deep_item_latent = tf.gather_nd(self.final_item_embedding[:,self.conf.total_len,:], self.eva_item_input) 

        eva_mf_user_latent = tf.gather_nd(self.shallow_user_embedding, self.eva_user_input)
        eva_mf_social_latent = tf.gather_nd(self.shallow_social_user_embedding, self.eva_user_input)
        eva_mf_item_latent = tf.gather_nd(self.shallow_item_embedding, self.eva_item_input)

        # --- EVA Feature combination, fusion Layer --- #
        # Left user1-user2 part
        #(deep)
        eva_user1_user2_basic_concat = tf.concat([eva_deep_basic_user1_latent, eva_deep_basic_user2_latent],1)
        eva_user1_user2_social_concat = tf.concat([eva_deep_social_user1_latent, eva_deep_social_user2_latent], 1)
        eva_total_user1_user2_left_concat = tf.concat([eva_user1_user2_social_concat, eva_user1_user2_basic_concat],1)
        #(shallow)
        eva_user1_user2_basic_multiply =tf.multiply(eva_user1_basic_mf_latent, eva_user2_basic_mf_latent)
        eva_user1_user2_social_multiply = tf.multiply(eva_mf_social_user1_latent, eva_mf_social_user2_latent) 
        eva_total_user1_user2_social_basic_multiply_concat_left = tf.concat([eva_user1_user2_social_multiply, eva_user1_user2_basic_multiply],1)


        # Right user-item part
        #(deep)
        eva_user_social_item_concat = tf.concat([eva_deep_social_latent, eva_deep_item_latent],1)
        eva_user_basic_item_concat = tf.concat([eva_deep_user_latent, eva_deep_item_latent],1)
        eva_total_user_item_concat = tf.concat([eva_user_social_item_concat, eva_user_basic_item_concat], 1)
        #(shallow)
        eva_user_social_item_multiply = tf.multiply(eva_mf_social_latent, eva_mf_item_latent)
        eva_user_basic_item_multiply = tf.multiply(eva_mf_user_latent, eva_mf_item_latent)
        eva_user_item_multiply = tf.concat([eva_user_social_item_multiply, eva_user_basic_item_multiply],1)


        # --- MLP LAYER --- #

        # SOCIAL
        
        if self.conf.layer_depth == 1:
            # 1 layer
            eva_s_mlp_output = self.social_first_MLP(eva_total_user1_user2_left_concat)

        if self.conf.layer_depth == 2:
            # 2 layers
            eva_s_mlp_output = self.social_second_MLP(self.social_first_MLP(eva_total_user1_user2_left_concat))

        if self.conf.layer_depth == 3:
            # 3 layers
            eva_s_mlp_output = self.social_third_MLP(self.social_second_MLP(self.social_first_MLP(eva_total_user1_user2_left_concat)))


        self.s_eva_prediction = self.y_s(tf.concat([eva_s_mlp_output, eva_total_user1_user2_social_basic_multiply_concat_left],1))


        # RATING
        if self.conf.layer_depth == 1:
            # 1 layer
            eva_r_mlp_output = self.rec_first_MLP(eva_total_user_item_concat)

        if self.conf.layer_depth == 2:
            # 2 layers
            eva_r_mlp_output = self.rec_second_MLP(self.rec_first_MLP(eva_total_user_item_concat))

        if self.conf.layer_depth == 3:
            # 3 layers
            eva_r_mlp_output = self.rec_third_MLP(self.rec_second_MLP(self.rec_first_MLP(eva_total_user_item_concat)))

        self.r_eva_prediction = self.y_r(tf.concat([eva_r_mlp_output,eva_user_item_multiply],1))        



        # --- LOSS PART --- #
        
        # RATING RMSE 
        self.rmse = tf.math.sqrt(\
                                tf.math.reduce_mean( (tf.math.square(self.labels_input - self.r_eva_prediction)),0 )
                                )
        
        # RATING LOSS 
        self.r_loss = tf.nn.l2_loss(self.labels_input - self.prediction) #+ self.conf.r_reg*(tf.nn.l2_loss(latest_user_latent) + tf.nn.l2_loss(latest_item_latent))
        self.test_loss = tf.nn.l2_loss(self.labels_input - self.r_eva_prediction)

        # SOCIAL LOSS 
        self.s_loss = tf.nn.l2_loss(self.s_labels_input - self.s_prediction) #+ self.conf.s_reg*(tf.nn.l2_loss(latest_user1_latent) + tf.nn.l2_loss(latest_user2_latent))
        #self.s_loss = tf.reduce_sum(tf.contrib.keras.losses.binary_crossentropy(self.s_labels_input, self.s_prediction))

        self.s_test_loss = tf.nn.l2_loss(self.s_labels_input - self.s_eva_prediction)
        #self.s_test_loss = tf.reduce_sum(tf.contrib.keras.losses.binary_crossentropy(self.s_labels_input, self.s_eva_prediction))

        # TOTAL LOSS
        #self.total_loss = cost1 + cost2
        #权重设置
        self.total_loss = self.s_loss + self.r_loss

        # OPTIMIZATION 
        #self.opt_loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.opt_loss = self.total_loss
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        #self.opt = tf.train.GradientDescentOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()


    
    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.temporal_user_embedding.op.name] = self.temporal_user_embedding
        variables_dict[self.temporal_item_embedding.op.name] = self.temporal_item_embedding

        '''
        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v
        '''
        #set_trace()
                
        #self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################
    
    
    
    def defineMap(self):
        map_dict = {}
        # RATING
        map_dict['r_train'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST',
            self.r_stamp_input: 'STAMP_LIST'
        }
        map_dict['r_test'] = {
            self.eva_user_input: 'TEST_USER_LIST', 
            self.eva_item_input: 'TEST_ITEM_LIST', 
            self.labels_input: 'TEST_LABEL_LIST'
        }

        map_dict['r_eva'] = {
            self.eva_user_input: 'EVA_USER_LIST', 
            self.eva_item_input: 'EVA_ITEM_LIST',
            #self.eva_labels_input: 'EVA_LABEL_LIST'
        }

        # SOCIAL
        map_dict['s_train'] = {
            self.user1_input: 'USER1_LIST', 
            self.user2_input: 'USER2_LIST', 
            self.s_labels_input: 'S_LABEL_LIST',
            self.s_stamp_input: 'S_STAMP_LIST'
        }
        map_dict['s_test'] = {
            self.eva_user1_input: 'TEST_USER1_LIST', 
            self.eva_user2_input: 'TEST_USER2_LIST', 
            self.s_labels_input: 'TEST_S_LABEL_LIST'
        }
        map_dict['s_eva'] = {
            self.eva_user1_input: 'EVA_USER1_LIST', 
            self.eva_user2_input: 'EVA_USER2_LIST',
            #self.eva_s_labels_input: 'EVAL_s_LABEL_LIST'

        }

        map_dict['out'] = {
            'r_train': self.r_loss,
            'r_test': self.test_loss,
            'r_eva': self.r_eva_prediction, 
            #'prediction': self.predict_vector,
            's_train': self.s_loss,
            's_test': self.s_test_loss,
            's_eva': self.s_eva_prediction, 
            #'s_prediction': self.s_predict_vector,
            'total_train_loss': self.total_loss,
            'rmse': self.rmse,
            's_loss': self.s_loss,
            'r_loss': self.r_loss,
            'userSocialOutputs': self.userSocialOutputs,
            'userSocialStates': self.userSocialStates,
            #'user': self.final_user_embedding,
            #'item': self.final_item_embedding,
            #'first_layer_ana': self.first_layer_analy, 
            #'second_layer_ana': self.second_layer_analy,
            #'first_layer_item_ana': self.first_layer_item_analy,
            #'second_layer_item_ana': self.second_layer_item_analy,
            #'stamp_output': self.stamp_input,
            #'stamp_output1': self.stamp_input1,
            'low_att_user_user': self.first_user_user_low_att,
            'low_att_user_item': self.first_user_item_low_att,
            'low_att_user_item_show': self.consumed_items_values_input1,
            'low_att_user_item_sparse_matrix': self.first_consumed_items_low_level_att_matrix
            #'low_att_user_user': self.first_item_user_low_att,
            #'first_social_neighbors_low_att_matrix': self.first_social_neighbors_low_level_att_matrix,
            #'second_social_neighbors_low_att_matrix': self.second_social_neighbors_low_level_att_matrix,
            #'first_consumed_items_low_level_att_matrix':self.first_consumed_items_low_level_att_matrix,
            #'second_consumed_items_low_level_att_matrix':self.second_consumed_items_low_level_att_matrix,
            #'first_items_users_neighborslow_level_att_matrix':self.first_items_users_neighborslow_level_att_matrix,
            #'second_items_users_neighborslow_level_att_matrix':self.second_items_users_neighborslow_level_att_matrix,

        }

        self.map_dict = map_dict







