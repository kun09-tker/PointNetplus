import os
import importlib
import modelnet_dataset
import numpy as np
import tensorflow as tf
from datetime import datetime

class PointNet():
    def __init__(self, num_classes, batch_size=32, num_point=1024, learning_rate=0.001, gpu_index = 0, momentum=0.9, optimizer = 'adam',decay_step=200000, decay_rate=0.7, max_epoch=251, train_dir = "", val_dir = "", ckpt_dir = ""):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_point = num_point
        self.learning_rate = learning_rate
        self.gpu_index = gpu_index
        self.momentum = momentum
        self.optimizer = optimizer
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.max_epoch = max_epoch
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.ckpt_dir = ckpt_dir
        self.model = importlib.import_module('pointnet2_cls_ssg')
        self.bn_init_decay = 0.5
        self.bn_decay_rate = 0.5
        self.bn_decay_step = float(self.decay_step)
        self.bn_decay_clip = 0.99
        self.train_dataset = modelnet_dataset.ModelNetDataset(self.train_dir, self.self.batch_size, self.self.num_point, shuffle=True)
        self.test_dataset = modelnet_dataset.ModelNetDataset(self.val_dir, self.self.batch_size, self.self.num_point, shuffle=False)
# self.batch_size = 16
# self.num_point = 40*17
# MAX_EPOCH = 251
# BASE_LEARNING_RATE = 0.001
# GPU_INDEX = 0
# MOMENTUM = 0.9
# OPTIMIZER = 'adam'
# DECAY_STEP = 200000
# DECAY_RATE = 0.7
# MODEL = importlib.import_module('pointnet2_cls_ssg')

# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)
# BN_DECAY_CLIP = 0.99

# NUM_CLASSES = 2

# self.train_dataset = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), self.batch_size=self.batch_size, npoints=self.num_point, shuffle=True)
# TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), self.batch_size=self.batch_size, npoints=self.num_point, shuffle=False)

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
                            self.learning_rate,  # Base learning rate.
                            batch * self.self.batch_size,  # Current index into the dataset.
                            self.decay_step,         # Decay step.
                            self.decay_rate,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        return learning_rate        

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
                        self.bn_init_decay, 
                        batch*self.self.batch_size,
                        self.bn_decay_step,
                        self.bn_decay_rate,
                        staircase=True)
        bn_decay = tf.minimum(self.bn_decay_clip, 1 - bn_momentum)
        return bn_decay

    def train_one_epoch(self, sess, ops):
        """ ops: dict mapping from string to tf ops """
        is_training = True
        
        # print(str(datetime.now()))

        # Make sure batch data is of same size
        cur_batch_data = np.zeros((self.self.batch_size,self.num_poin,self.train_dataset.num_channel()))
        cur_batch_label = np.zeros((self.self.batch_size), dtype=np.int32)

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        batch_idx = 0
        while self.train_dataset.has_next_batch():
            batch_data, batch_label = self.train_dataset.next_batch(augment=True)
            #batch_data = provider.random_point_dropout(batch_data)
            bsize = batch_data.shape[0]
            cur_batch_data[0:bsize,...] = batch_data
            cur_batch_label[0:bsize] = batch_label

            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                        ops['labels_pl']: cur_batch_label,
                        ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            # train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
            total_correct += correct
            total_seen += bsize
            loss_sum += loss_val
            if (batch_idx+1)%50 == 0:
                print(' ---- batch: %03d ----' % (batch_idx+1))
                print('mean loss: %f' % (loss_sum / 50))
                print('accuracy: %f' % (total_correct / float(total_seen)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
            batch_idx += 1

        self.train_dataset.reset()
            
    def eval_one_epoch(self, sess, ops):
        """ ops: dict mapping from string to tf ops """
        global EPOCH_CNT
        is_training = False

        # Make sure batch data is of same size
        self.test_dataset
        cur_batch_data = np.zeros((self.batch_size, self.num_point, self.test_dataset.num_channel()))
        cur_batch_label = np.zeros((self.batch_size), dtype=np.int32)

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        batch_idx = 0
        shape_ious = []
        total_seen_class = [0 for _ in range(self.num_classes)]
        total_correct_class = [0 for _ in range(self.num_classes)] 
        
        print(str(datetime.now()))
        print('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
        
        while self.test_dataset.has_next_batch():
            batch_data, batch_label = self.test_dataset.next_batch(augment=False)
            bsize = batch_data.shape[0]
            # for the last batch in the epoch, the bsize:end are from last batch
            cur_batch_data[0:bsize,...] = batch_data
            cur_batch_label[0:bsize] = batch_label

            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                        ops['labels_pl']: cur_batch_label,
                        ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            # test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
            total_correct += correct
            total_seen += bsize
            loss_sum += loss_val
            batch_idx += 1
            for i in range(0, bsize):
                l = batch_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)
        
        print('eval mean loss: %f' % (loss_sum / float(batch_idx)))
        print('eval accuracy: %f'% (total_correct / float(total_seen)))
        print('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
        EPOCH_CNT += 1

        self.test_dataset.reset()
        return total_correct/float(total_seen)

    def train(self):
        """ Simple training function """
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(self.gpu_index)):
                pointclouds_pl, labels_pl = self.model.placeholder_inputs(self.batch_size, self.num_point)
                is_training_pl = tf.placeholder(tf.bool, shape=())
                
                # Note the global_step=batch parameter to minimize. 
                # That tells the optimizer to helpfully increment the 'batch' parameter
                # for you every time it trains.
                batch = tf.get_variable('batch', [],
                    initializer=tf.constant_initializer(0), trainable=False)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss 
                pred, end_points = self.model.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
                self.model.get_loss(pred, labels_pl, end_points)
                losses = tf.get_collection('losses')
                total_loss = tf.add_n(losses, name='total_loss')
                tf.summary.scalar('total_loss', total_loss)
                for l in losses + [total_loss]:
                    tf.summary.scalar(l.op.name, l)

                correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self.batch_size)
                tf.summary.scalar('accuracy', accuracy)

                print("--- Get training operator")
                # Get training operator
                learning_rate = self.get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if self.optimizer == 'momentum': 
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.momentum)
                elif self.optimizer == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(total_loss, global_step=batch)
                
                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()
            
            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Add summary writers
            # merged = tf.summary.merge_all()
            # train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
            # test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ops = {'pointclouds_pl': pointclouds_pl,
                'labels_pl': labels_pl,
                'is_training_pl': is_training_pl,
                'pred': pred,
                'loss': total_loss,
                'train_op': train_op,
                'step': batch,
                'end_points': end_points}

            best_acc = -1
            for epoch in range(self.max_epoch):
                print('**** EPOCH %03d ****' % (epoch))
                # sys.stdout.flush()
                
                self.train_one_epoch(sess, ops)
                self.eval_one_epoch(sess, ops)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(self.ckpt_dir, "model.ckpt"))
                    print("Model saved in file: %s" % save_path)