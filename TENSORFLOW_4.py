# NIHARIKA PENTAPATI
# PES1201700215


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("model_data/", one_hot=True)

import tensorflow as tf

logs_path = '/tmp/tensorflow_logs/'
batch = 100
learning_rate = 0.01
training_epochs = 20
h1_neurons = 128
h2_neurons = 256

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.random_normal([784, h1_neurons]))
b1 = tf.Variable(tf.zeros([h1_neurons]))
W2 = tf.Variable(tf.random_normal([h1_neurons,h2_neurons]))
b2 = tf.Variable(tf.zeros([h2_neurons]))
W3 = tf.Variable(tf.random_normal([h2_neurons,10]))
b3 = tf.Variable(tf.zeros([10]))

h1 = tf.nn.relu(tf.matmul(x,W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)
y = tf.matmul(h2,W3) + b3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    merged = tf.summary.merge_all()

    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples / batch)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch)
            sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})
        if epoch % 2 == 0:
            print ("Epoch : ", epoch)
            print ("Train accuracy : ", accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
            
    print("Training completed!")
    print("Test accuracy : ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

