### Import Libraries

import tensorflow as tf  ### import Tensorflow
from tensorflow.examples.tutorials.mnist import input_data  ### Import DataSet
#from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn


def Recurrent_Neural_Network(data, classes, batch_size, pic_chunk,no_chunks,Size_of_rnn, epochs):
	rnn_layer = {'weight' : tf.Variable(tf.random_normal([Size_of_rnn, classes])),
					'bias' : tf.Variable(tf.random_normal([classes]))}   ###  Dictionary of weight and Bias for layer 1

   	data = tf.transpose(data, [1,0,2])
	data = tf.reshape(data, [-1, pic_chunk])
	data = tf.split(data, no_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(Size_of_rnn)

	outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)

	output = tf.matmul(outputs[-1], rnn_layer['weight']) + rnn_layer['bias']



	return output

def train_model(data,features,labels, classes, batch_size, pic_chunk,no_chunks,Size_of_rnn, epochs):
	prediction = Recurrent_Neural_Network(features, classes, batch_size, pic_chunk,no_chunks,Size_of_rnn, epochs)   ### Call our Model for Training
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= prediction, labels=labels))   ### Loss function / Cost Function


	optimizer = tf.train.AdamOptimizer().minimize(loss)   ### Optimizer to Optimize the Loss

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			loss_per_epoch = 0   ### container for our updated Loss/Cost

			for x in range(int(data.train.num_examples/batch_size)):   ### Iterate through each Training Example
				features_epochs , labels_epochs = data.train.next_batch(batch_size)
				features_epochs = features_epochs.reshape((batch_size, no_chunks, pic_chunk))

				optimize, cost = sess.run([optimizer, loss], feed_dict={features: features_epochs, labels:labels_epochs})   ### feeding the values of Features and Labels
				loss_per_epoch = loss_per_epoch + cost

			print('Epochs', epoch, 'Loss', loss_per_epoch)

		compare = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)) 
		
		accuracy = tf.reduce_mean(tf.cast(compare, 'float'))

		print('Accuracy', accuracy.eval({features: data.test.images.reshape((-1,no_chunks, pic_chunk)), labels: data.test.labels}))


def read_data():
	mnist = input_data.read_data_sets('MNiST_data', one_hot=True)  ### Read MNIST Data

	return mnist

def main():

	pic_chunk = 28   ### 28x28 pixel
	no_chunks = 28
	Size_of_rnn = 128   ### LSTM Size (Number of cells)
	classes = 10   ### Total classes (0-9)
	batch_size = 100  ### Chunk Size(batch Processing)
	epochs = 20   ### Number of Iterations  (Feed Forward with Back Propogation)

	### Matric of Height and width
	features = tf.placeholder('float',[None, no_chunks, pic_chunk])  ### Input Data
	labels = tf.placeholder('float', )  ### Actual Output

	data = read_data()   ### Call for Data Read Funtion
	train_model(data,features,labels, classes, batch_size,pic_chunk,no_chunks,Size_of_rnn, epochs)


if __name__ == '__main__':  ### Starting Point of the Program
	main()
