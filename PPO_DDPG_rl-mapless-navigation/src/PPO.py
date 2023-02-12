from encodings import utf_8
from struct import pack

from ipykernel import kernel_protocol_version
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
import os
from tensorflow.keras.layers import Input, Dense, BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
import tensorflow as tf




import time
from copy import deepcopy
import numpy as np
import pickle as p

NOISE = 1.0 # Exploration noise
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, 2)), np.zeros((1, 1))
REPLAY_START_SIZE = 10000
class Memory:
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_s_ = []
        self.batch_done = []

    def store(self, s, a, s_, r, done):
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)

    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()

    @property
    def cnt_samples(self):
        return len(self.batch_s)


class Agent:
    
  
    def __init__(self, dic_agent_conf, dic_path, dic_env_conf):
        
        self.train=False
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path
        self.dic_env_conf = dic_env_conf
        self.time_step = 0
        self.n_actions = self.dic_agent_conf["ACTION_DIM"]
        self.load=False
        self.actor_network = self._build_actor_network()
        self.actor_old_network = self.build_network_from_copy(self.actor_network)

        self.critic_network = self._build_critic_network()

        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self.n_actions))
       
        #self.load_model()
        self.memory = Memory()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray), "state must be numpy.ndarry"

        state = np.reshape(state, [-1, 16])
        prob = self.actor_network.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
       
        """action=[]
        action[0] =np.random.choice(1, p=prob[0])
        action[1] =np.random.choice(1, p=prob[1])"""
        action =prob + np.random.normal(loc=0, scale=NOISE, size=prob.shape)
        return action

    def train_network(self):
        n = self.memory.cnt_samples
        discounted_r = []
        if self.memory.batch_done[-1]:
            v = 0
        else:
            v = self.get_v(self.memory.batch_s_[-1])
        for r in self.memory.batch_r[::-1]:
            v = r + self.dic_agent_conf["GAMMA"] * v
            discounted_r.append(v)
        discounted_r.reverse()

        batch_s, batch_a, batch_discounted_r = np.vstack(self.memory.batch_s), \
                     np.vstack(self.memory.batch_a), \
                     np.vstack(discounted_r)

        batch_v = self.get_v(batch_s)
        batch_advantage = batch_discounted_r - batch_v
        batch_old_prediction = self.get_old_prediction(batch_s)

        batch_a_final = np.zeros(shape=(len(batch_a), self.n_actions))
        #batch_a_final[:, batch_a.flatten()] = 1
        
        # print(batch_s.shape, batch_advantage.shape, batch_old_prediction.shape, batch_a_final.shape)
        ator_path=os.path.join(self.dic_path["PATH_TO_MODEL_actor"], "%s_actor_network" % self.time_step)
        critic_path=os.path.join(self.dic_path["PATH_TO_MODEL_actor"], "%s_critic_network" % self.time_step)
        #checkpointa = ModelCheckpoint(ator_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
        #checkpointc = ModelCheckpoint(critic_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)

        
        chis=self.critic_network.fit(x=batch_s, y=batch_discounted_r,epochs=4, verbose=0)
        ahis=self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction],epochs=4,  y=np.array(batch_a), verbose=0)
    
        
        
        f = open("demofileactor2.csv", "a")
        fc = open("demofilecritic2.csv", "a")

       

        listToStr = ' \t'.join(map(str, ahis.history['loss']))
        f.write(listToStr)
        

        listToStr = ' \t'.join(map(str, chis.history['loss']))
        fc.write(listToStr)
        

        f.write(" \n\n\n")
        fc.write(" \n\n\n")
        fc.close()
        f.close()
        self.memory.clear()
        self.update_target_network()

    def get_old_prediction(self, s):

        s = np.reshape(s,(-1, 16))
        return self.actor_old_network.predict_on_batch([s, DUMMY_VALUE, DUMMY_ACTION ])

    def store_transition(self, s, a, r, s_, done):
        self.memory.store(s, a, s_, r, done)

    def get_v(self, s):
        s = np.reshape(s, (-1, self.dic_agent_conf["STATE_DIM"]))
        v = self.critic_network.predict_on_batch(s)
        return v

    def save_model(self, file_name):
        
        save_path_a=self.dic_path["PATH_TO_MODEL_actor"]
        save_path_c=self.dic_path["PATH_TO_MODEL_critic"]
        #self.actor_network.save(os.path.join(self.dic_path["PATH_TO_MODEL_actor"], "%s_actor_network.h5" % file_name))
        #self.critic_network.save(os.path.join(self.dic_path["PATH_TO_MODEL_critic"],"%s_critic_network.h5" % file_name))
       
        #np.save(os.path.join(save_path_c, "%s.h5" % file_name), self.critic_network.optimizer.get_weights()) 
        #np.save(os.path.join(save_path_a, "%s.h5" % file_name), self.actor_network.optimizer.get_weights()) 

        opt_path_a=os.path.join("/home/radia/catkin_ws/rl-mapless-navigation/src/trained_models/ppo/actor/opt","%s.pkl"%file_name)

        opt_path_c=os.path.join("/home/radia/catkin_ws/rl-mapless-navigation/src/trained_models/ppo/critic/opt","%s.pkl"%file_name)


        self.actor_network.save_weights(os.path.join(save_path_a,'%s.h5'% file_name))
        symbolic_weights = getattr(self.actor_network.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(opt_path_a, 'wb') as f:
        
            p.dump(weight_values, f)


        self.critic_network.save_weights(os.path.join(save_path_c,'%s.h5'% file_name))
        symbolic_weights_c = getattr(self.critic_network.optimizer, 'weights')
        weight_values_c = K.batch_get_value(symbolic_weights_c)
        with open(opt_path_c, 'wb') as c:
        
            p.dump(weight_values_c, c)



      


    def load_model_actor(self,model,name,advantage,old_prediction):   
      
        #optimizer = tf.keras.optimizers.Adam()
        #if len(dir) > 0:
        print("Successfully loaded  network")
        load_path=os.path.join(self.dic_path["PATH_TO_MODEL_actor"], "%s.h5" % name)
        opt_path=os.path.join("/home/radia/catkin_ws/rl-mapless-navigation/src/trained_models/ppo/actor/opt","%s.pkl"%name)

        model.compile(optimizer='adam', loss=self.proximal_policy_optimization_loss(
                                    advantage=advantage, old_prediction=old_prediction,
                                ))
        model.load_weights(load_path) #.h5
        model._make_train_function()
        with open(opt_path, 'rb') as f:
            weight_values = p.load(f)
        model.optimizer.set_weights(weight_values)
        
        #K.clear_session()
        #actor_network = load_model(os.path.join(self.dic_path["PATH_TO_MODEL_actor"], "860_actor_network.h5"),compile=False)#, custom_objects={'loss': self.proximal_policy_optimization_loss()})#os.path.join(self.dic_path["PATH_TO_MODEL_actor"],"160_actor_network"))
        #actor_network=self.load_optimizer_state(actor_network, load_path, "860", actor_network.trainable_weights)
        

        """opt_weights = np.load(os.path.join(load_path, "860")+'.npy', allow_pickle=True) 
        zero_grads = [tf.zeros_like(w) for w in actor_network.trainable_weights] 
        saved_vars = [tf.identity(w) for w in actor_network.trainable_weights]
        optimizer.apply_gradients(zip(zero_grads, actor_network.trainable_weights)) 
        [x.assign(y) for x,y in zip(actor_network.trainable_weights, saved_vars)]
        optimizer.set_weights(opt_weights)"""
        
        print("radia")
        return model
            
    
    def load_model_critic(self,model, name):   
        optimizer = tf.keras.optimizers.Adam()
        load_path=os.path.join(self.dic_path["PATH_TO_MODEL_critic"], "%s.h5" % name)
        opt_path=os.path.join("/home/radia/catkin_ws/rl-mapless-navigation/src/trained_models/ppo/critic/opt","%s.pkl"%name)

        print("Successfully loaded  network")
        model.compile(optimizer='adam', loss=self.dic_agent_conf["CRITIC_LOSS"])
        model.load_weights(load_path) #.h5
        model._make_train_function()
        with open(opt_path, 'rb') as f:
            weight_values = p.load(f)
        model.optimizer.set_weights(weight_values)
        #if len(dir) > 0:
        print("Successfully loaded  network")
        #K.clear_session()
        #critic_network = load_model(os.path.join(self.dic_path["PATH_TO_MODEL_critic"], "860_critic_network.h5"),compile=False)#, custom_objects={'loss': self.proximal_policy_optimization_loss()})#os.path.join(self.dic_path["PATH_TO_MODEL_critic"], "160_critic_network"))
        #self.actor_old_network = deepcopy(self.actor_network)
        #critic_network=self.load_optimizer_state(critic_network, load_path, "860", critic_network.trainable_weights)

 
        """opt_weights = np.load(os.path.join(load_path, "860")+'.npy', allow_pickle=True) 
        zero_grads = [tf.zeros_like(w) for w in critic_network.trainable_weights] 
        saved_vars = [tf.identity(w) for w in critic_network.trainable_weights]
        optimizer.apply_gradients(zip(zero_grads, critic_network.trainable_weights)) 
        [x.assign(y) for x,y in zip(critic_network.trainable_weights, saved_vars)]
        optimizer.set_weights(opt_weights)"""
        return model
       
            
    def _build_actor_network(self):

        state = Input(shape=(self.dic_agent_conf["STATE_DIM"],), name="state")

        advantage = Input(shape=(1, ), name="Advantage")
        old_prediction = Input(shape=(self.n_actions,), name="Old_Prediction")

        shared_hidden = self._shared_network_structure(state)

        action_dim = self.dic_agent_conf["ACTION_DIM"]

        policy = Dense(action_dim, activation="tanh", name="actor_output_layer")(shared_hidden)
        policy =BatchNormalization()(policy) 

        dir = os.listdir(self.dic_path["PATH_TO_MODEL_actor"])
        actor_network = Model(inputs=[state, advantage, old_prediction], outputs=policy)
        if self.load:
            actor_network=self.load_model_actor(actor_network,"515000",advantage,old_prediction)
            #for i in range( len(actor_network.layers)):
            #  actor_network.layers[i].set_weights(actor_network2.layers[i].get_weights())
            
        else:
            optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"])
        
            if self.dic_agent_conf["OPTIMIZER"] is "Adam":
                actor_network.compile(optimizer=optimizer,#Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                    loss=self.proximal_policy_optimization_loss(
                                        advantage=advantage, old_prediction=old_prediction,
                                    ))


        print("=== Build Actor Network ===")
        actor_network.summary()

        time.sleep(1.0)
        return actor_network

    def update_target_network(self):
        alpha = self.dic_agent_conf["TARGET_UPDATE_ALPHA"]
        self.actor_old_network.set_weights(alpha*np.array(self.actor_network.get_weights())
                                           + (1-alpha)*np.array(self.actor_old_network.get_weights()))

                        

    def _build_critic_network(self):
        state = Input(shape=(self.dic_agent_conf["STATE_DIM"],), name="state")
        shared_hidden = self._shared_network_structure(state)

        if self.dic_env_conf["POSITIVE_REWARD"]:
            q = Dense(1, activation="relu", name="critic_output_layer")(shared_hidden)
            q =BatchNormalization()(q) 
        else:
            q = Dense(1, name="critic_output_layer")(shared_hidden)
            q =BatchNormalization()(q) 


        dir = os.listdir(self.dic_path["PATH_TO_MODEL_critic"])
        critic_network = Model(inputs=state, outputs=q)
        if self.load:
            critic_network=self.load_model_critic(critic_network,"515000")
        else:
            optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"])
    
            

            if self.dic_agent_conf["OPTIMIZER"] is "Adam":
                critic_network.compile(optimizer=optimizer,#Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                    loss=self.dic_agent_conf["CRITIC_LOSS"])



        print("=== Build Critic Network ===")
        critic_network.summary()

        time.sleep(1.0)
        return critic_network

    def build_network_from_copy(self, actor_network):
        network_structure = actor_network.to_json()
        network_weights = actor_network.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]), loss="mse")
        return network

    def _shared_network_structure(self, state_features):
        #dense_d = self.dic_agent_conf["D_DENSE"]
        hidden1 = Dense(512, activation="relu", name="hidden_shared_1")(state_features)
        hidden1 =BatchNormalization()(hidden1) 
        
        hidden2 = Dense(512, activation="relu", name="hidden_shared_2")(hidden1)
        hidden2 =BatchNormalization()(hidden2) 

        hidden3 = Dense(512, activation="relu", name="hidden_shared_3")(hidden2)
        hidden3 =BatchNormalization()(hidden3)
        return hidden3

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        LOSS_CLIPPING = self.dic_agent_conf["CLIPPING_LOSS_RATIO"]
        """entropy_loss = self.dic_agent_conf["ENTROPY_LOSS_RATIO"]

        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                           max_value=1 + loss_clipping) * advantage) + entropy_loss * (
                           prob * K.log(prob + 1e-10)))"""

        def loss(y_true, y_pred):
            var = K.square(NOISE)
            pi = 3.1415926
            denom = K.sqrt(2 * pi * var)
            prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

            prob = prob_num/denom
            old_prob = old_prob_num/denom
            r = prob/(old_prob + 1e-10)

            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))

        return loss



    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.store_transition(state,action,reward,next_state,done)
        if len(self.memory.batch_a)== REPLAY_START_SIZE :
            print('\n---------------Start training---------------')
            self.train=True
        #if len(self.memory.batch_a)> REPLAY_START_SIZE :
        if self.train==True:
            self.time_step += 1
            self.train_network()
        
        if self.time_step % 5000 == 0 and self.time_step > 0:
            self.save_model(str(self.time_step))

        return self.time_step


    ''' Save keras.optimizers object state. Arguments: optimizer --- Optimizer object.
          save_path --- Path to save location. save_name --- Name of the .npy file to be created.
          ''' # Create folder if it does not exists 

    def save_optimizer_state(optimizer, save_path, save_name):
       
        if not os.path.exists(save_path): 
            os.makedirs(save_path) # save weights 
            np.save(os.path.join(save_path, save_name), optimizer.get_weights()) 
        return



    ''' Loads keras.optimizers object state. Arguments: optimizer --- Optimizer object to be loaded. load_path -
         -- Path to save location. load_name --- Name of the .npy file to be read. model_train_vars ---
          List of model variables (obtained using Model.trainable_variables) ''' 
          # Load optimizer weights 
    def load_optimizer_state(net, load_path, load_name, model_train_vars):
        
        opt_weights = np.load(os.path.join(load_path, load_name)+'.npy', allow_pickle=True) 
        # dummy zero gradients 
        zero_grads = [tf.zeros_like(w) for w in model_train_vars] 
        # save current state of variables 
        saved_vars = [tf.identity(w) for w in model_train_vars]
        # Apply gradients which don't do nothing with Adam 
        net.optimizer.apply_gradients(zip(zero_grads, model_train_vars)) 
        # Reload variables 
        [x.assign(y) for x,y in zip(model_train_vars, saved_vars)] 
        # # Set the weights of the optimizer 
        net.optimizer.set_weights(opt_weights) 
        return net