#NAME:NIHARIKA PENTAPATI
#SRN:PES1201700215

import tensorflow as tf
import csv

f = open('train.csv','r')
data = list(csv.reader(f))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(1.5)
b = tf.Variable(0.)
y1 = (W*x)+b

error = 0.5*tf.reduce_mean(tf.square(y1-y))
opt = tf.train.GradientDescentOptimizer(0.0005).minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	x_data,y_data = [float(i[0]) for i in data[1:]],[float(i[1]) for i in data[1:]]
	for j in range(20):
		for i in range(len(x_data)):
			sess.run(opt,{x:x_data,y:y_data})
	m = sess.run(W)
	c = sess.run(b)
	print("Calculated Equation --> y = ",m,"x",c,sep="")
	error_final = sess.run(error,feed_dict = {x:x_data,y:y_data})
	print("Final error",error_final)
	
