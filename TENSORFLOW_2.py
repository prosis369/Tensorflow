# Author: Niharika Pentapati
# SRN: PES1201700215


import tensorflow as tf

user_data = input().split()
new_data = [float(i) for i in user_data]

x_data = []
first_row = []
second_row = []
third_row = []
y_data = []

for i in range(0, len(new_data)):
	
	if(i<3):
		first_row.append(new_data[i])
	elif(i>3 and i<7):
		second_row.append(new_data[i])
	elif(i>7 and i<11):
		third_row.append(new_data[i])
	else:
		y_data.append([new_data[i]])
x_data.append(first_row)
x_data.append(second_row)
x_data.append(third_row)

x = tf.placeholder("float",None)
y = tf.placeholder("float",None)
a = tf.matrix_inverse(x)
result = tf.matmul(a,y)

with tf.Session() as sess:
	print(sess.run(result,feed_dict={x:x_data,y:y_data}))
	
