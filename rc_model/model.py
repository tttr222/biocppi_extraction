import sys, os, random, pickle, re, time
import numpy as np
import tensorflow as tf
import sklearn.metrics as skm
import sklearn.model_selection as skmodel

class WordCNN(object):
    def __init__(self, labels, word_vocab,
                    max_sequence_length,
                    word_embeddings=None,
                    filter_sizes=[3,4,5], 
                    num_filters=200,
                    embedding_size=200,
                    optimizer='default', learning_rate='default', 
                    embedding_factor = 1.0, decay_rate=1.0, 
                    dropout_keep=0.5, 
                    num_cores=5,
                    rnn_unit='lstm'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.inter_op_parallelism_threads=num_cores
        config.intra_op_parallelism_threads=num_cores 
        
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)
        self.labels = []
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        self.optimizer = optimizer
        self.decay = decay_rate
        
        if optimizer == 'default':
            self.optimizer = 'rmsprop'
        else:
            self.optimizer = optimizer
        
        if learning_rate is not 'default':
            self.lrate = float(learning_rate)
        else:
            if self.optimizer in ['adam','rmsprop']:
                self.lrate = 0.001
            elif self.optimizer == 'adagrad':
                self.lrate = 0.5
            else:
                raise Exception('Unknown optimizer {}'.format(optimizer))
        
        print >> sys.stderr, "Optimizer: {}, Learning rate: {}, Decay rate: {}".format(
            self.optimizer, self.lrate, self.decay)
        
        self.embedding_factor = embedding_factor
        self.dropout_keep = dropout_keep
        self.labels = labels
        self.word_vocab = word_vocab
        self.word_embeddings = word_embeddings
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.rnn_dim = 200
        self.rnn_unit = rnn_unit
        
        self._compile()
    
    def _compile(self):
        output_size = len(self.labels)
        
        #batch size
        bsize = None
        seqlen = self.max_sequence_length

        # input and output
        self.y_true = tf.placeholder(tf.float32, [bsize, len(self.labels)])
        self.x_input = tf.placeholder(tf.int32, [bsize, seqlen])
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0),shape=None)
        
        # -----------------------------------------------------------------------------
        # embeddings
        W_em = tf.Variable(tf.truncated_normal([len(self.word_vocab), self.embedding_size], 
                    stddev=1.0/np.sqrt(seqlen)))
        self.w_input = tf.placeholder(tf.float32, [len(self.word_vocab), self.embedding_size])
        self.embedding_init = W_em.assign(self.w_input)
        
        xw_input = tf.expand_dims(tf.nn.embedding_lookup(W_em, self.x_input), -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(xw_input, W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(h,
                            ksize=[1, seqlen - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        cnn_pool = tf.concat(pooled_outputs,axis=3)
        cnn_pool_flat = tf.nn.dropout(tf.reshape(cnn_pool, [-1, num_filters_total]), self.keep_prob)
        
        W_out = tf.Variable(tf.truncated_normal([num_filters_total, len(self.labels)], 
                    stddev=1.0/np.sqrt(num_filters_total)))
        b_out = tf.Variable(tf.zeros([len(self.labels)]))
        self.y_out = tf.matmul(cnn_pool_flat, W_out) + b_out
        
        self.y_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y_out,labels=self.y_true))

        self.y_prob = tf.nn.softmax(self.y_out)
        self.y_true_idx = tf.argmax(self.y_true,axis=1)
        self.y_pred_idx = tf.argmax(self.y_prob,axis=1)
        
        self.y_pred = tf.one_hot(self.y_pred_idx,depth=output_size,dtype=tf.int32)
        
        epoch_step = tf.Variable(0, trainable=False)
        self.epoch_step_op = tf.assign(epoch_step, epoch_step+1)
        
        lrate_decay = tf.train.exponential_decay(self.lrate, epoch_step, 1, self.decay)
            
        if self.optimizer == 'adam':
            self.train_step = tf.train.AdamOptimizer(learning_rate=lrate_decay).minimize(self.y_loss)
        elif self.optimizer == 'adagrad':
            self.train_step = tf.train.AdagradOptimizer(learning_rate=lrate_decay,
                                                initial_accumulator_value=1e-08).minimize(self.y_loss)
        elif self.optimizer == 'rmsprop':
            self.train_step = tf.train.RMSPropOptimizer(learning_rate=lrate_decay,epsilon=1e-08).minimize(self.y_loss)
        else:
            raise Exception(('The optimizer {} is not in list of available ' 
                            + 'optimizers: default, adam, adagrad, rmsprop.')
                            .format(self.optimizer))
        
        self.sess.run(tf.global_variables_initializer())
        if self.word_embeddings is not None:
            print "Initializing word embeddings matrix with pretained values.."
            self.sess.run(self.embedding_init, { self.w_input: self.word_embeddings})
            
        self.saver = tf.train.Saver(max_to_keep=100)
        print >> sys.stderr, "Compiled model with feature vector of length {}".format(num_filters_total)
    
    def _onehot(self, y, categories):
        y_onehot = np.zeros((len(y),len(categories)))
        for i in range(len(y)):
            y_onehot[i,categories.index(y[i])] = 1
        
        return y_onehot
    
    def _build_feed_dict(self,batch,dropout=True):
        #x_data, y_true = zip(*batch)
        #x_train, f_train, h_train, u_train  = zip(*x_data)
        feed_dict = {}
        
        X_feed = []
        y_feed = []
        maxlen = 0
        zero_vec = self.word_vocab.index('<ZERO>')
        unk_vec = self.word_vocab.index('UNK')
        for xx, yy in batch:
            _, _, _, text = xx
            xin = [ zero_vec ] * 4
            for s in text:
                for w in s:
                    if w.lower() in self.word_vocab:
                        xin.append(self.word_vocab.index(w.lower()))
                    else:
                        xin.append(unk_vec)
                    
                xin += [ zero_vec ] * 4
            
            maxlen = max(len(xin),maxlen)
            X_feed.append(xin)
            y_feed.append(yy)
    
        #print "Found maxlen of", maxlen
        X_feed = [ xx + [0] * (self.max_sequence_length - len(xx)) for xx in X_feed ]
        
        feed_dict[self.x_input] = X_feed
        feed_dict[self.y_true] = y_feed
        
        if dropout:
            feed_dict[self.keep_prob] = self.dropout_keep
        
        return feed_dict
    
    def _train_minibatches(self,minibatches):
        mavg_loss = None
        
        for k, minibatch in enumerate(minibatches):
            varl = [self.train_step, self.y_loss, self.y_pred_idx, self.y_true_idx]
            _, ym_loss, ym_pred, ym_true = self.sess.run(varl, minibatch)
            
            if mavg_loss is None:
                mavg_loss = ym_loss
            else:
                mavg_loss = 0.8 * mavg_loss + 0.2 * ym_loss
            
            sys.stdout.write(" >> training {}/{} loss={:.7f}  \r".format(
                k+1,len(minibatches),mavg_loss))
            sys.stdout.flush()
    
    def fit(self, X, y, num_epoch = 10, batch_size = 32, seed = 1, 
                validation_split = 0.20, overfit_training = False):
                    
        random.seed(seed)
        job_hash = time.time()
        trainset = zip(X,self._onehot(y,self.labels))
        print "Target labels: {}".format(self.labels)
        
        trainset = random.sample(trainset,len(trainset))
        
        if overfit_training:
            train_split, valid_split = trainset, trainset[:200]
        else:
            if isinstance(validation_split,float):
                m = int(len(y) * validation_split)
            else:
                m = validation_split
            
            train_split, valid_split = trainset[m:], trainset[:m]
            
            assert(len(train_split) + m == len(y))
            assert(len(valid_split) == m)
    
        print "{}/{} in training/validation set".format(len(train_split),len(valid_split))
        trainsp = train_split[:200]
        trainfd = self._build_feed_dict(trainsp,dropout=False)
        valfd = self._build_feed_dict(valid_split,dropout=False)
        
        best_epoch = 0       
        best_score = 0
        best_model = None
            
        for i in range(1,num_epoch+1):
            estart = time.time()
            batchpool = random.sample(train_split,len(train_split))
            
            minibatches = []
            for k in range(0,len(batchpool),batch_size):
                pool = batchpool[k:k+batch_size]
                minibatches.append(self._build_feed_dict(pool))
            
            self._train_minibatches(minibatches)
            self.sess.run(self.epoch_step_op)
            
            loss, yt_pred, yt_true = self.sess.run([self.y_loss, self.y_pred_idx, self.y_true_idx], trainfd)
            f, _, _ = self.fscore(yt_pred,yt_true)
            
            vloss, yv_pred, yv_true = self.sess.run([self.y_loss, self.y_pred_idx, self.y_true_idx], valfd)
            vf, vprecision, vrecall = self.fscore(yv_pred,yv_true)

            save_marker = ''
            if vf >= best_score:
                best_model = './scratch/model-{}-t{}-e{}-s{}.ckpt'.format(
                    type(self).__name__.lower(), job_hash,i,seed)
                
                best_epoch, best_score = i, vf
                self.saver.save(self.sess, best_model)
                save_marker = '*'
                
            elapsed = int(time.time() - estart)
            emin, esec = elapsed / 60, elapsed % 60
            #print "epoch {} loss {:.5f} fit {:.2%} vloss {:.5f} fit {:.2%} [{}m{}s] {}".format(i, 
            #    loss, acc, vloss, vacc, emin, esec, save_marker)
            print "epoch {} bsize={} loss {:.5f} fit {:.2%} val {:.2%}/{:.2%}/{:.2%}[{}m{}s] {}".format(i, 
                batch_size, loss, f, vf, vprecision, vrecall, emin, esec, save_marker)
                
        
        if best_model is None:
            print "WARNING: NO GOOD FIT"
        
        self.saver.restore(self.sess, best_model)
        print "Fitted to model from epoch {} with score {} at {}".format(best_epoch,best_score,best_model)
    
    def save(self, model_path):
        self.saver.save(self.sess, model_path)
    
    def restore(self, model_path):
        tf.reset_default_graph()
        self.saver.restore(self.sess, model_path)
    
    def predict(self, X, batch_size = 100):
        dummy_labels = [self.labels[0]] * len(X)
        dummy_y = self._onehot(dummy_labels,self.labels)
        testset_all = zip(X,dummy_y)
        
        prediction_idx = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self._build_feed_dict(testset,dropout=False)
            prediction_idx += list(self.sess.run(self.y_pred_idx, testfd))
        
        return [ self.labels[idx] for idx in prediction_idx ]
    
    def predict_proba(self, X, batch_size = 100):
        dummy_labels = [self.labels[0]] * len(X)
        dummy_y = self._onehot(dummy_labels,self.labels)
        testset_all = zip(X,dummy_y)
        
        y_prob_list = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self._build_feed_dict(testset,dropout=False)
            y_prob_list.append(self.sess.run(self.y_prob, testfd))
        
        return np.concatenate(y_prob_list,axis=0)

    def evaluate(self,X,y, batch_size = 100, macro = False):
        testset_all = zip(X,self._onehot(y,self.labels))
        
        y_pred_idx = []
        y_true_idx = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self._build_feed_dict(testset,dropout=False)
            yp, yt = self.sess.run([self.y_pred_idx,self.y_true_idx], testfd)
            y_pred_idx += list(yp)
            y_true_idx += list(yt)
    
        return self.fscore(y_pred_idx,y_true_idx)
    
    def fscore(self,y_pred,y_true, macro=False):
        avg_meth = 'micro'
        labels = [ i for i in range(len(self.labels)) if self.labels[i] not in ['Negative'] ]
        f = skm.fbeta_score(y_true, y_pred, average=avg_meth,labels=labels, beta=1)
        p = skm.precision_score(y_true, y_pred, average=avg_meth,labels=labels)
        r = skm.recall_score(y_true, y_pred, average=avg_meth,labels=labels)
        return f, p ,r


