#encoding=utf-8
import numpy as np
import tensorflow as tf

def dataSet():
	f = open('data.txt', 'r')
	data = np.zeros((15060, 14))
	label = np.zeros((15060, 1))

	for i in range(15060):
		row = f.readline().split()
		#print data[i,:].shape
		data[i,:] = row[0:14]
		label[i, 0] = 0 if row[14] is '1' else 1
	return data, label	

def normal(data):
	temp = np.sum(data, 0)
	for i in range(14):
		data[:,i] = data[:,i]/temp[i]
	return data


def lr(x,w,b):
	return 1.0/(1+tf.exp(-(tf.add(tf.matmul(x,w),b))))

def tarin_model(train_x, train_y, epoch = 10000, rate = 0.1):
	train_x = np.array(train_x)
	train_y = np.array(train_y)
	n = train_x.shape[0]
	x = tf.placeholder("float", [n, 14])
	y = tf.placeholder('float')
	w = tf.Variable(tf.random_normal([14, 1]))
	b = tf.Variable(tf.random_normal([n, 1]))

	pred = lr(x, w, b)
	loss = tf.reduce_sum(tf.pow(pred- y, 2))
	optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss) 

	init = tf.initialize_all_variables()
	sess =tf.Session()
	
	sess.run(init)
	for i in range(epoch):
		sess.run(optimizer,{x:train_x,y:train_y})
		if i%100 == 0:
			print sess.run(loss, {x:train_x,y:train_y})
	w =  sess.run(w)
	b = sess.run(b)
	return w,b

def forward(test_x, test_y, w, b):
	W = tf.placeholder(tf.float32)
	B = tf.placeholder(tf.float32)
	X = tf.placeholder(tf.float32)
	Y = tf.placeholder(tf.float32)
	pred = lr(X, W, B)
	loss = tf.reduce_mean(tf.pow(pred-Y,2))
	sess = tf.Session()
	
	print '==================='
	print sess.run(pred, {X:test_x,Y:test_y,W:w,B:b})

	loss = sess.run(loss,{X:test_x,Y:test_y,W:w,B:b}) #get all prediction loss
	return loss



if __name__ == "__main__":
	train_x,train_y = dataSet()
	train_x = normal(train_x)
	print train_x
	w,b = tarin_model(train_x,train_y)
	print 'weights',w
	print 'bias',b
	loss = forward(train_x,train_y,w,b)
	print loss
	print train_y
