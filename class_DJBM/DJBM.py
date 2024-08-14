from __future__ import division
import tensorflow as tf
import numpy as np
from ipdb import set_trace
from tensorflow.contrib.rnn.python.ops import rnn_cell
import keras

class DJBM():
    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'ITEM_CUSTOMER_SPARSE_MATRIX'
        )

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()




    def inputSupply(self, data_dict):
        low_att_std = 1.0
        #social neighbors
        #self.first_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_SN_layer1')
        #self.first_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_SN_layer2')

        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']

        #self.social_neighbors_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_social_neighbors_layer1( \
        #                                    tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=low_att_std)),[-1,1])      )   ),1)

        #first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_values_input1, axes=0)
        #self.first_user_user_low_att = [first_mean_social_influ, first_var_social_influ]



        #self.second_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_SN_layer1')
        #self.second_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_SN_layer2')
        #self.social_neighbors_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_social_neighbors_layer1( \
        #                                    tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=1.0)),[-1,1])      )   ),1)


        #self.social_neighbors_values_input3 = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=0.01))
        #self.social_neighbors_num_input = 1.0/np.reshape(data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'],[-1,1])



        # user-item

        # low-att layers
        #self.first_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_UI_layer1')
        #self.first_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_UI_layer2')

        #self.user_item_sparsity_dict = data_dict['USER_ITEM_SPARSITY_DICT']  #Sparsity

        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']

        #self.consumed_items_values_input1 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        #self.consumed_items_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_user_item_layer1( \
        #                                    tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=low_att_std)),[-1,1])  )   ),1)
        
        #first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.consumed_items_values_input1, axes=0)
        #self.first_user_item_low_att = [first_mean_social_influ, first_var_social_influ]


        #self.second_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_UI_layer1')
        #self.second_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_UI_layer2')
        #self.consumed_items_values_input2 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        #self.consumed_items_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_user_item_layer1( \
        #                                    tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=1.0)),[-1,1])    )   ),1)

        #self.consumed_items_values_input3 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        #self.consumed_items_num_input = 1.0/np.reshape(data_dict['CONSUMED_ITEMS_NUM_INPUT'], [-1,1])



        # item-user

        #self.first_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_IU_layer1')
        #self.first_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_IU_layer2')


        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']
        self.item_customer_values_input = data_dict['ITEM_CUSTOMER_VALUES_INPUT']
        #self.item_customer_values_input1 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        #self.item_customer_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_item_user_layer1( \
        #                                    tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=low_att_std)),[-1,1])    )   ),1)


        #first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.item_customer_values_input1, axes=0)
        #self.first_item_user_low_att = [first_mean_social_influ, first_var_social_influ]


        #self.second_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_IU_layer1')
        #self.second_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_IU_layer2')
        #self.item_customer_values_input2 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        #self.item_customer_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_item_user_layer1( \
        #                                    tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01)),[-1,1]) )   ),1)

        #self.item_customer_values_input3 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        #self.item_customer_num_input = 1.0/np.reshape(data_dict['ITEM_CUSTOMER_NUM_INPUT'],[-1,1])

        

        # prepare sparse matrix, in order to compute user's embedding from social neighbors and consumed items
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.item_customer_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)





        ######## Add High Level Attention Here ########

        '''

        self.first_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "first_layer_user_part_influence_attention")
        first_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.first_layer_user_attention_ini),1)
        self.first_layer_user_attention = tf.div(tf.math.exp(self.first_layer_user_attention_ini), first_layer_user_attention_norm_denominator)

        self.first_user_userneighbor_attention_value = tf.slice(self.first_layer_user_attention,[0,0],[1,1])
        self.first_user_itemneighbor_attention_value = tf.slice(self.first_layer_user_attention,[0,1],[1,1])

        # Second Layer Influence user Part:
        self.second_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "second_layer_user_part_influence_attention")
        second_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.second_layer_user_attention_ini),1)
        second_layer_user_attention = tf.div(tf.math.exp(self.second_layer_user_attention_ini), second_layer_user_attention_norm_denominator)

        self.second_user_userneighbor_attention_value = tf.slice(second_layer_user_attention,[0,0],[1,1])
        self.second_user_itemneighbor_attention_value = tf.slice(second_layer_user_attention,[0,1],[1,1])        

        # Third Layer Influence user Part:
        self.third_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "third_layer_user_part_influence_attention")
        third_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.third_layer_user_attention_ini),1)
        third_layer_user_attention = tf.div(tf.math.exp(self.third_layer_user_attention_ini), third_layer_user_attention_norm_denominator)

        self.third_user_userneighbor_attention_value = tf.slice(third_layer_user_attention,[0,0],[1,1])
        self.third_user_itemneighbor_attention_value = tf.slice(third_layer_user_attention,[0,1],[1,1])     





        # First Layer Influence Item Part:
        self.first_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "first_layer_item_part_influence_attention")
        first_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.first_layer_item_attention_ini),1)
        self.first_layer_item_attention = tf.div(tf.math.exp(self.first_layer_item_attention_ini), first_layer_item_attention_norm_denominator)

        self.first_item_itself_attention_value = tf.slice(self.first_layer_item_attention,[0,0],[1,1])
        self.first_item_userneighbor_attention_value = tf.slice(self.first_layer_item_attention,[0,1],[1,1])

        # Second Layer Influence Item Part:
        self.second_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "second_layer_item_part_influence_attention")
        second_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.second_layer_item_attention_ini),1)
        second_layer_item_attention = tf.div(tf.math.exp(self.second_layer_item_attention_ini), second_layer_item_attention_norm_denominator)

        self.second_item_itself_attention_value = tf.slice(second_layer_item_attention,[0,0],[1,1])
        self.second_item_userneighbor_attention_value = tf.slice(second_layer_item_attention,[0,1],[1,1])        

        # Third Layer Influence Item Part:
        self.third_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "third_layer_item_part_influence_attention")
        third_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.third_layer_item_attention_ini),1)
        third_layer_item_attention = tf.div(tf.math.exp(self.third_layer_item_attention_ini), third_layer_item_attention_norm_denominator)

        self.third_item_itself_attention_value = tf.slice(third_layer_item_attention,[0,0],[1,1])
        self.third_item_userneighbor_attention_value = tf.slice(third_layer_item_attention,[0,1],[1,1])      

        '''


        ######## Add Low Level Attention Here #########
        # First layer low att or avg

        ##### Avg_version #####
        

        
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

        


        # low att

        '''
        self.first_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input1,
            dense_shape=self.social_neighbors_dense_shape
        )
        '''

        '''
        self.first_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input1,
            dense_shape=self.consumed_items_dense_shape
        )
        '''
       
        '''
        self.first_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input1,
            dense_shape=self.item_customer_dense_shape
        )
        '''

        
        '''
        self.first_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.first_layer_social_neighbors_sparse_matrix) 
        #set_trace()
        self.first_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.first_layer_consumed_items_sparse_matrix) 
        self.first_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.first_layer_item_customer_sparse_matrix) 
        '''



        # Second layer 

        # avg
        '''
        self.second_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        '''

        self.second_layer_social_neighbors_sparse_matrix = {}
        self.second_layer_consumed_items_sparse_matrix = {}
        self.second_layer_item_customer_sparse_matrix = {}
        for t in range(self.conf.total_len):
            self.second_layer_social_neighbors_sparse_matrix[t] = tf.SparseTensor(
                indices = self.social_neighbors_indices_input[t], 
                values = self.social_neighbors_values_input[t],
                dense_shape=self.social_neighbors_dense_shape
            )
            self.second_layer_consumed_items_sparse_matrix[t] = tf.SparseTensor(
                indices = self.consumed_items_indices_input[t], 
                values = self.consumed_items_values_input[t],
                dense_shape=self.consumed_items_dense_shape
            )

            
            self.second_layer_item_customer_sparse_matrix[t] = tf.SparseTensor(
                indices = self.item_customer_indices_input[t], 
                values = self.item_customer_values_input[t],
                dense_shape=self.item_customer_dense_shape
            )


        #set_trace()
        # low att
        '''
        self.second_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input2,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.second_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input2,
            dense_shape=self.consumed_items_dense_shape
        )
        self.second_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input2,
            dense_shape=self.item_customer_dense_shape
        )

        
        self.second_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.second_layer_social_neighbors_sparse_matrix) 
        self.second_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.second_layer_consumed_items_sparse_matrix) 
        self.second_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.second_layer_item_customer_sparse_matrix) 
        
        '''


        
        # Third layer low att



        # avg

        '''
        self.third_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        '''

        '''
        self.third_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input,
            dense_shape=self.consumed_items_dense_shape
        )
        self.third_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input,
            dense_shape=self.item_customer_dense_shape
        )

        '''


        # low att

        '''
        self.third_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input3,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.third_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input3,
            dense_shape=self.consumed_items_dense_shape
        )
        self.third_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input3,
            dense_shape=self.item_customer_dense_shape
        )

        self.third_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.third_layer_social_neighbors_sparse_matrix)
        self.third_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.third_layer_consumed_items_sparse_matrix)
        self.third_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.third_layer_item_customer_sparse_matrix)
        '''

    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y


    #  Matrix Mul  First Layer 


    # AVG

    
    def generateUserEmbeddingFromSocialNeighbors_avg(self, current_user_embedding):
        user_embedding_from_social_neighbors = []
        for t in range(self.conf.total_len):
            user_embedding_from_social_neighbors.append(tf.sparse_tensor_dense_matmul(self.social_neighbors_sparse_matrix_avg[t], current_user_embedding[t]))

        return tf.reshape(user_embedding_from_social_neighbors,[self.conf.total_len, self.conf.num_users, self.conf.dimension])
   

    def generateUserEmebddingFromConsumedItems_avg(self, current_item_embedding):
        user_embedding_from_consumed_items = []
        for t in range(self.conf.total_len):
            user_embedding_from_consumed_items.append(tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix_avg[t], current_item_embedding[t]))


        return tf.reshape(user_embedding_from_consumed_items,[self.conf.total_len,self.conf.num_users,self.conf.dimension])

    def generateItemEmebddingFromCustomer_avg(self, current_user_embedding):
        item_embedding_from_customer = []
        for t in range(self.conf.total_len):
            item_embedding_from_customer.append(tf.sparse_tensor_dense_matmul(self.item_customer_sparse_matrix_avg[t], current_user_embedding[t]))
        return tf.reshape(item_embedding_from_customer, [self.conf.total_len,self.conf.num_items,self.conf.dimension])    



    # low_att

    '''
    def generateUserEmbeddingFromSocialNeighbors1(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.first_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    '''




    '''
    def generateUserEmebddingFromConsumedItems1(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.first_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items
    '''



    '''
    def generateItemEmebddingFromCustomer1(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.first_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer
    '''


    #  Matrix Mul  Second Layer 

    # avg
    '''
    def generateUserEmbeddingFromSocialNeighbors_avg(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix_avg, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    '''

    '''
    def generateUserEmebddingFromConsumedItems_avg(self, current_item_embedding):
        user_embedding_from_consumed_items = {}
        for t in range(11):
            user_embedding_from_consumed_items[t] = tf.sparse_tensor_dense_matmul(
                self.consumed_items_sparse_matrix_avg[t], current_item_embedding[t]
            )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer_avg(self, current_user_embedding):
        item_embedding_from_customer = {}
        for t in range(11):
            item_embedding_from_customer[t] = tf.sparse_tensor_dense_matmul(
                self.item_customer_sparse_matrix_avg[t], current_user_embedding[t]
            )
        return item_embedding_from_customer
    '''

    # low ATT
    ''' 
    def generateUserEmbeddingFromSocialNeighbors2(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.second_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    


    
    def generateUserEmebddingFromConsumedItems2(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.second_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer2(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.second_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer

    '''

    #  Matrix Mul  Third Layer 

    #avg 

    '''
    def generateUserEmbeddingFromSocialNeighbors_avg(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix_avg, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    
    def generateUserEmebddingFromConsumedItems_avg(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.consumed_items_sparse_matrix_avg, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer_avg(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.item_customer_sparse_matrix_avg, current_user_embedding
        )
        return item_embedding_from_customer
    '''

    # low att
    '''
    def generateUserEmbeddingFromSocialNeighbors3(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.third_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    
    def generateUserEmebddingFromConsumedItems3(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.third_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer3(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.third_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer
    '''



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
        self.user_social_temporal_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.total_len, self.conf.dimension], stddev=stddev_std), name='user_embedding')
        self.user_temporal_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.total_len, self.conf.dimension], stddev=stddev_std), name='user_embedding')
        self.item_temporal_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.total_len, self.conf.dimension], stddev=stddev_std), name='item_embedding')

        self.user_social_ini_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, 1, self.conf.dimension], stddev=stddev_std), name='user_social_ini_embedding')
        self.user_ini_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, 1, self.conf.dimension], stddev=stddev_std), name='user_ini_embedding')
        self.item_ini_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, 1, self.conf.dimension], stddev=stddev_std), name='item_ini_embedding')

        # This part for the tranditional MF
        self.shallow_user_embedding = tf.Variable(tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=stddev_std), name='shallow_user_embedding')
        self.shallow_item_embedding = tf.Variable(tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=stddev_std), name='shallow_item_embedding')
        self.shallow_social_user_embedding = tf.Variable(tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=stddev_std), name='shallow_social_user_embedding')


        self.reduce_dimension_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='reduce_dimension_layer')


        # High-ATT
        #set_trace()
        self.first_user_part_social_high_att_layer1 = {}
        self.first_user_part_interest_high_att_layer1 = {}
        for t in range(self.conf.total_len):
            self.first_user_part_social_high_att_layer1[t] = tf.layers.Dense(1, activation=tf.nn.tanh, name='usu1'+str(t)+'1')
            self.first_user_part_interest_high_att_layer1[t] = tf.layers.Dense(1, activation=tf.nn.tanh, name='uiu'+str(t)+'1')

        self.first_user_part_social_high_att_layer2 = tf.layers.Dense(1, activation=tf.nn.tanh, name='1')

       

        '''
        # User part
        self.first_user_part_social_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='1')

        self.first_user_part_social_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='2')

        self.first_user_part_interest_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='3')

        self.first_user_part_interest_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='4')


        self.second_user_part_social_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='5')

        self.second_user_part_social_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='6')

        self.second_user_part_interest_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='7')

        self.second_user_part_interest_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='8')


        # Item part
        self.first_item_part_itself_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='9')

        self.first_item_part_itself_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='10')

        self.first_item_part_user_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='11')

        self.first_item_part_user_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='12')




        self.second_item_part_itself_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='13')

        self.second_item_part_itself_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='14')


        self.second_item_part_user_high_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='15')

        self.second_item_part_user_high_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='16')


        '''


        '''
        self.att_layer_item = tf.layers.Dense(\
            1, activation=tf.nn.sigmoid, name='attention_layer')

        self.transformation = tf.layers.Dense(\
            64, activation=tf.nn.relu, name='attention_layer')

        self.item_fusion_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='item_fusion_layer')
        self.user_fusion_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='user_fusion_layer')
        '''

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
        self.temporal_item_embedding = tf.transpose(self.item_temporal_embedding, [1, 0, 2])   # t v d



        #self.item_embedding = tf.Variable(tf.random_normal([self.conf.num_items+1, self.conf.dimension], stddev=0.01), name='item_test_embedding')

        # First Layer
        T_first_user_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors_avg(self.temporal_user_social_embedding)

        T_first_user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems_avg(self.temporal_item_embedding)
        
        T_first_item_embedding_from_customer = self.generateItemEmebddingFromCustomer_avg(self.temporal_user_embedding)
        
        # Second Layer
        #T_second_user_social_embedding_from_social_neighbors = self.generateUserEmebddingFromConsumedItems_avg(T_first_user_embedding_from_social_neighbors)
        #T_second_user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems_avg(T_first_item_embedding_from_customer)
        #T_second_item_embedding_from_customer = self.generateItemEmebddingFromCustomer_avg(T_first_user_embedding_from_consumed_items)


        user_social_last_embedding = self.temporal_user_social_embedding + T_first_user_embedding_from_social_neighbors \
                                     #+ T_second_user_social_embedding_from_social_neighbors

        user_last_embedding = self.temporal_user_embedding + T_first_user_embedding_from_consumed_items \
                              #+ T_second_user_embedding_from_consumed_items

        item_last_embedding = self.temporal_item_embedding + T_first_item_embedding_from_customer \
                              #+ T_second_item_embedding_from_customer                              


        #user_social_last_embedding_new = user_last_embedding + user_social_last_embedding



        # GRU for social user embedding evolution 
        with tf.variable_scope("social_user_rnn_cell"):
            userSocialCell = tf.nn.rnn_cell.GRUCell(num_units=self.conf.dimension)
            #userSocialCell = rnn_cell.LayerNormBasicLSTMCell(num_units=self.conf.dimension)
            #userInput = user_last_embedding   # [t_step, user_num, dim]
            userSocialInput = tf.transpose(user_social_last_embedding, [1, 0, 2])   # because dynamic_rnn needs time_len as the second input
            userSocialOutputs, userSocialStates = tf.nn.dynamic_rnn(userSocialCell, userSocialInput, dtype=tf.float32, time_major=False)
            self.userSocialOutputs = userSocialOutputs  # output is a multi dimension array
            self.userSocialStates = userSocialStates    # the last state of the output

        # GRU for preference user embedding evolution 
        with tf.variable_scope("user_rnn_cell"):
            userCell = tf.nn.rnn_cell.GRUCell(num_units=self.conf.dimension)
            #userCell = rnn_cell.LayerNormBasicLSTMCell(num_units=self.conf.dimension)
            #userInput = user_last_embedding   # [t_step, user_num, dim]
            userInput = tf.transpose(user_last_embedding, [1, 0, 2])
            userOutputs, userStates = tf.nn.dynamic_rnn(userCell, userInput, dtype=tf.float32, time_major=False)
            self.userOutputs = userOutputs
            self.userStates = userStates


        # GRU for item embedding evolution
        with tf.variable_scope("item_rnn_cell"):
            itemCell = tf.nn.rnn_cell.GRUCell(num_units=self.conf.dimension)
            #itemCell = rnn_cell.LayerNormBasicLSTMCell(num_units=self.conf.dimension)

            #itemInput = item_last_embedding   # [t_step, item_num, dim]
            itemInput = tf.transpose(item_last_embedding, [1, 0, 2])
            itemOutputs, itemStates = tf.nn.dynamic_rnn(itemCell, itemInput, dtype=tf.float32, time_major=False)
            self.itemOutputs = itemOutputs
            self.itemStates = itemStates


        #set_trace() 
        #aaa= self.first_user_part_social_high_att_layer2(self.itemOutputs)


        #self.userSocialOutputs = self.userSocialOutputs + self.userOutputs
        #self.userSocialStates = self.userSocialStates + self.userStates


        

        user_preference_att={}
        user_social_att={}
        #self.user_preference_att1={}
        #self.user_social_att1={}
        attsum = {}


        # ATT part

        '''  
        for t in range(self.conf.total_len):
            user_preference_att[t] = tf.math.exp(self.first_user_part_interest_high_att_layer1[t](self.userOutputs[:,t,:]))
            user_social_att[t] = tf.math.exp(self.first_user_part_social_high_att_layer1[t](self.userSocialOutputs[:,t,:]))
            attsum[t] =  user_preference_att[t] + user_social_att[t]
            user_preference_att[t] = user_preference_att[t] / attsum[t]
            user_social_att[t] = user_social_att[t] / attsum[t]

            if t==0:
                user_social_att1 = tf.reshape((user_social_att[t] * self.userSocialOutputs[:,t,:]),[1,self.conf.num_users,self.conf.dimension])
                user_preference_att1 = tf.reshape((user_preference_att[t] * self.userOutputs[:,t,:]),[1,self.conf.num_users,self.conf.dimension])
            elif t>0:
                user_social_att1 = tf.concat([user_social_att1,tf.reshape((user_social_att[t] * self.userSocialOutputs[:,t,:]),[1,self.conf.num_users,self.conf.dimension])],0)
                user_preference_att1 = tf.concat([user_preference_att1,tf.reshape((user_preference_att[t] * self.userOutputs[:,t,:]),[1,self.conf.num_users,self.conf.dimension])],0)

            self.user_social_att1 = user_social_att1
            self.user_preference_att1 = user_preference_att1

        self.user_social_att1 = tf.transpose(self.user_social_att1,[1,0,2])
        self.user_preference_att1 = tf.transpose(self.user_preference_att1,[1,0,2])



        self.userSocialOutputs = self.user_social_att1*self.userSocialOutputs
        self.userOutputs = self.user_preference_att1 * self.userOutputs
        self.userSocialStates = self.userSocialOutputs[:,self.conf.total_len-1,:]
        self.userStates = self.userOutputs[:,self.conf.total_len-1,:]
        
	'''
        
        '''
        self.userSocialOutputs = self.user_social_att1*self.userSocialOutputs + self.user_preference_att1*self.userOutputs
        self.userSocialStates = self.userSocialOutputs[:,self.conf.total_len-1,:]
        '''

        '''
        self.userSocialOutputs = self.user_social_att1*self.userSocialOutputs + self.user_preference_att1*self.userOutputs
        self.userSocialStates = self.userSocialOutputs[:,self.conf.total_len-1,:]
        '''


        # concat the stamp 0        
        # [user_num, 11+1, dim]
        self.final_social_user_embedding = tf.concat([self.user_social_ini_embedding, self.userSocialOutputs], 1)
        # [user_num, 11+1, dim]
        #self.final_user_embedding = tf.concat([self.user_ini_embedding, self.userOutputs], 1)
        self.final_user_embedding = tf.concat([self.user_ini_embedding, self.userOutputs], 1)
        # [item_num, 11+1, dim]
        self.final_item_embedding = tf.concat([self.item_ini_embedding, self.itemOutputs], 1) 
        


        #--- Embedding Extracting --- #
        # social part
        user1_social_stamp_indices=tf.concat([tf.reshape(self.user1_input,[-1,1]),tf.reshape(self.s_stamp_input,[-1,1])],1)
        user2_social_stamp_indices=tf.concat([tf.reshape(self.user2_input,[-1,1]),tf.reshape(self.s_stamp_input,[-1,1])],1)
        
        deep_social_user1_latent = tf.gather_nd(self.final_social_user_embedding, user1_social_stamp_indices)
        deep_social_user2_latent = tf.gather_nd(self.final_social_user_embedding, user2_social_stamp_indices)

        mf_social_user1_latent = tf.gather_nd(self.shallow_social_user_embedding, self.user1_input)
        mf_social_user2_latent = tf.gather_nd(self.shallow_social_user_embedding, self.user2_input)


        deep_basic_user1_latent = tf.gather_nd(self.final_user_embedding, user1_social_stamp_indices)
        deep_basic_user2_latent = tf.gather_nd(self.final_user_embedding, user2_social_stamp_indices)

        user1_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.user1_input)
        user2_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.user2_input)

        # rating part
        user_stamp_indices=tf.concat([tf.reshape(self.user_input,[-1,1]),tf.reshape(self.r_stamp_input,[-1,1])],1)
        item_stamp_indices=tf.concat([tf.reshape(self.item_input,[-1,1]),tf.reshape(self.r_stamp_input,[-1,1])],1)

        deep_user_latent = tf.gather_nd(self.final_user_embedding, user_stamp_indices)  
        deep_social_latent = tf.gather_nd(self.final_social_user_embedding, user_stamp_indices)  
        deep_item_latent = tf.gather_nd(self.final_item_embedding, item_stamp_indices) 

        mf_user_latent = tf.gather_nd(self.shallow_user_embedding, self.user_input)
        mf_social_latent = tf.gather_nd(self.shallow_social_user_embedding, self.user_input)
        mf_item_latent = tf.gather_nd(self.shallow_item_embedding, self.item_input)


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
        s_mlp_output = self.social_third_MLP(self.social_second_MLP(self.social_first_MLP(total_user1_user2_left_concat)))
        self.s_prediction = self.y_s(tf.concat([s_mlp_output,total_user1_user2_social_basic_multiply_concat_left],1))

        #rating prediction
        r_mlp_output = self.rec_third_MLP(self.rec_second_MLP(self.rec_first_MLP(total_user_item_concat)))
        self.prediction = self.y_r(tf.concat([r_mlp_output,user_item_multiply],1))        

     
     


        # --- EVA PART --- #

        #eva_deep_social_user1_embedding = tf.gather_nd(self.userSocialStates, self.user1_input)
        #eva_deep_social_user2_embedding = tf.gather_nd(self.userSocialStates, self.user2_input)


        # social part
        eva_deep_social_user1_latent = tf.gather_nd(self.userSocialStates, self.eva_user1_input) #use the last stamp for testing
        eva_deep_social_user2_latent = tf.gather_nd(self.userSocialStates, self.eva_user2_input)

        eva_mf_social_user1_latent = tf.gather_nd(self.shallow_social_user_embedding, self.eva_user1_input)
        eva_mf_social_user2_latent = tf.gather_nd(self.shallow_social_user_embedding, self.eva_user2_input)


        eva_deep_basic_user1_latent = tf.gather_nd(self.userStates, self.eva_user1_input)
        eva_deep_basic_user2_latent = tf.gather_nd(self.userStates, self.eva_user2_input)

        eva_user1_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.eva_user1_input)
        eva_user2_basic_mf_latent = tf.gather_nd(self.shallow_user_embedding, self.eva_user2_input)

        # rating part
        eva_deep_user_latent = tf.gather_nd(self.userStates, self.eva_user_input)  
        eva_deep_social_latent = tf.gather_nd(self.userSocialStates, self.eva_user_input)  
        eva_deep_item_latent = tf.gather_nd(self.itemStates, self.eva_item_input) 

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
        eva_s_mlp_output = self.social_third_MLP(self.social_second_MLP(self.social_first_MLP(eva_total_user1_user2_left_concat)))
        self.s_eva_prediction = self.y_s(tf.concat([eva_s_mlp_output, eva_total_user1_user2_social_basic_multiply_concat_left],1))

        eva_r_mlp_output = self.rec_third_MLP(self.rec_second_MLP(self.rec_first_MLP(eva_total_user_item_concat)))
        self.r_eva_prediction = self.y_r(tf.concat([eva_r_mlp_output,eva_user_item_multiply],1))        







     
        # --- Prediction Layer --- #

        # RATING
        #self.predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        #self.prediction = tf.sigmoid(tf.reduce_sum(self.predict_vector, 1, keepdims=True))

        
        #self.eva_predict_vector = tf.multiply(eva_user_embedding, eva_item_embedding)
        #self.r_eva_prediction = tf.sigmoid(tf.reduce_sum(self.eva_predict_vector, 1, keepdims=True))
        # SOCIAL
        #self.s_predict_vector = tf.multiply(latest_user1_latent, latest_user2_latent)
        #self.s_prediction = tf.sigmoid(tf.reduce_sum(self.s_predict_vector, 1, keepdims=True))

        #self.s_eva_predict_vector = tf.multiply(eva_deep_social_user1_embedding, eva_deep_social_user2_embedding)
        #self.s_eva_prediction = tf.sigmoid(tf.reduce_sum(self.s_eva_predict_vector, 1, keepdims=True))
        




        # LOSS PART

        # RATING RMSE 
 
        self.rmse = tf.math.sqrt(\
                                tf.math.reduce_mean( (tf.math.square(self.labels_input - self.r_eva_prediction)),0 \
                                )
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
        self.total_loss = self.s_loss + 10*self.r_loss

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
            'r_loss': self.r_loss
            #'user': self.final_user_embedding,
            #'item': self.final_item_embedding,
            #'first_layer_ana': self.first_layer_analy, 
            #'second_layer_ana': self.second_layer_analy,
            #'first_layer_item_ana': self.first_layer_item_analy,
            #'second_layer_item_ana': self.second_layer_item_analy,
            #'stamp_output': self.stamp_input,
            #'stamp_output1': self.stamp_input1,
            #'low_att_user_user': self.first_user_user_low_att,
            #'low_att_user_item': self.first_user_item_low_att,
            #'low_att_user_user': self.first_item_user_low_att,
            #'first_social_neighbors_low_att_matrix': self.first_social_neighbors_low_level_att_matrix,
            #'second_social_neighbors_low_att_matrix': self.second_social_neighbors_low_level_att_matrix,
            #'first_consumed_items_low_level_att_matrix':self.first_consumed_items_low_level_att_matrix,
            #'second_consumed_items_low_level_att_matrix':self.second_consumed_items_low_level_att_matrix,
            #'first_items_users_neighborslow_level_att_matrix':self.first_items_users_neighborslow_level_att_matrix,
            #'second_items_users_neighborslow_level_att_matrix':self.second_items_users_neighborslow_level_att_matrix,

        }

        self.map_dict = map_dict



















