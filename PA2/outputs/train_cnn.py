import time


import tensorflow as tf

initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Weights creation
    def create_weight(self, shape):
        return tf.get_variable("W", shape=shape, initializer=initializer)

    # Bias creation
    def create_bias(self, shape):
        return tf.get_variable("b", shape=shape,initializer=initializer)

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        # =====================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

        w = tf.get_variable("W1", shape=[784, hidden_size], initializer=initializer)
        b = tf.get_variable("b1", shape=[hidden_size], initializer=initializer)
        h = tf.sigmoid(tf.matmul(X, w) + b)
        return h


    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

        conv1 = tf.layers.conv2d(inputs=X,filters=40,kernel_size=5,activation=tf.nn.sigmoid)
                
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
        
        conv2 = tf.layers.conv2d(inputs=pool1,filters=40,kernel_size=5,activation=tf.nn.sigmoid)
        
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

        pool2 = tf.reshape(pool2,[-1,640])

        w = tf.get_variable("W", shape=[640, hidden_size], initializer=initializer)
        b = tf.get_variable("b", shape=[hidden_size], initializer=initializer)
        
        h =  tf.sigmoid(tf.matmul(pool2,w) + b)

        return h

    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

        conv1 = tf.layers.conv2d(inputs=X,filters=40,kernel_size=5,activation=tf.nn.relu)
                
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
        
        conv2 = tf.layers.conv2d(inputs=pool1,filters=40,kernel_size=5,activation=tf.nn.relu)
        
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

        pool2 = tf.reshape(pool2,[-1,640])

        w = tf.get_variable("W", shape=[640, hidden_size], initializer=initializer)
        b = tf.get_variable("b", shape=[hidden_size], initializer=initializer)
        
        h =  tf.nn.relu(tf.matmul(pool2,w) + b)

        return h

    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

        conv1 = tf.layers.conv2d(inputs=X,filters=40,kernel_size=5,activation=tf.nn.relu)
                
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
        
        conv2 = tf.layers.conv2d(inputs=pool1,filters=40,kernel_size=5,activation=tf.nn.relu)
        
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

        pool2 = tf.reshape(pool2,[-1,640])

        w = tf.get_variable("W", shape=[640, hidden_size], initializer=initializer, regularizer=regularizer)
        b = tf.get_variable("b", shape=[hidden_size], initializer=initializer)
        
        h1 =  tf.nn.relu(tf.matmul(pool2,w) + b)

        w1 = tf.get_variable("W1", shape=[hidden_size, hidden_size], initializer=initializer, regularizer=regularizer)
        b1 = tf.get_variable("b1", shape=[hidden_size], initializer=initializer)

        h = tf.nn.relu(tf.matmul(h1,w1) + b1)

        return h

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)


        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            #X = tf.placeholder(tf.float32, [None, 784])
            Y = tf.placeholder(tf.int32, [None])
            Y = tf.one_hot(Y, 10)
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)

            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)

            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to logits after code implementation.
            #logits = NotImplementedError


            ''' 
            with tf.name_scope("softmax"):
                w_o = tf.get_variable("w_o", shape=[hidden_size, class_num], initializer=initializer)
                b_o = tf.get_variable("b_o", shape=[class_num], initializer=initializer)
                logit = tf.matmul(features, w_o) + b_o
            '''


            with tf.name_scope("softmax"):
                w_o = tf.get_variable("w_o", shape=[hidden_size, class_num], initializer=initializer)
                b_o = tf.get_variable("b_o", shape=[class_num], initializer=initializer)
                logit = tf.matmul(features, w_o) + b_o


            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to loss after code implementation.
            #loss = NotImplementedError

            with tf.name_scope("loss"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logit))

            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to train_op after code implementation.
            #train_op = NotImplementedError

            with tf.name_scope("optimizer"):
                train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            #
            #accuracy = NotImplementedError

            with tf.name_scope("accuracy"):
                label = Y
                correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(logit,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            
            has_GPU = False
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))

                    total_correct = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    print ('accuracy of the trained model %f' % (total_correct))# / testX.shape[0]))
                    print ()

                return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False}) #/ testX.shape[0]