### Import Libraries

import tensorflow as tf  ### import Tensorflow
from tensorflow.examples.tutorials.mnist import input_data  ### Import DataSet

def Deep_Neural_Network(data, node_hidden_layer_1, node_hidden_layer_2, node_hidden_layer_3, classes, batch_size):
	hidden_layer_1 = {'weight' : tf.Variable(tf.random_normal([784, node_hidden_layer_1])),
					  'bias' : tf.Variable(tf.random_normal([node_hidden_layer_1]))}   ###  Dictionary of weight and Bias for layer 1

	hidden_layer_2 = {'weight' : tf.Variable(tf.random_normal([node_hidden_layer_1, node_hidden_layer_2])),
					  'bias' : tf.Variable(tf.random_normal([node_hidden_layer_2]))}   ###  Dictionary of weight and Bias for layer 2
	
	hidden_layer_3 = {'weight' : tf.Variable(tf.random_normal([node_hidden_layer_2, node_hidden_layer_3])),
					  'bias' : tf.Variable(tf.random_normal([node_hidden_layer_3]))}   ###  Dictionary of weight and Bias for layer 3

	output_layer = {'weight' : tf.Variable(tf.random_normal([node_hidden_layer_3, classes])),
					 'bias' : tf.Variable(tf.random_normal([classes]))}   ###  Dictionary of weight and Bias for output layer

	layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weight']), hidden_layer_1['bias'])   ### Input data * weights * Bias (Model)
	layer_1 = tf.nn.relu(layer_1)   ### Threshold Function wether the Neurons Fires or not

	layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weight']), hidden_layer_2['bias'])   ### Input data * weights * Bias (Model)
	layer_2 = tf.nn.relu(layer_2)   ### Threshold Function wether the Neurons Fires or not
	
	layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weight']) , hidden_layer_3['bias'])   ### Input data * weights * Bias (Model)
	layer_3 = tf.nn.relu(layer_3)   ### Threshold Function wether the Neurons Fires or not

	output = tf.matmul(layer_3, output_layer['weight']) + output_layer['bias']

	return output

def train_model(data,features,labels, node_hidden_layer_1, node_hidden_layer_2, node_hidden_layer_3, classes, batch_size):
	prediction = Deep_Neural_Network(features, node_hidden_layer_1, node_hidden_layer_2, node_hidden_layer_3, classes, batch_size)   ### Call our Model for Training
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= prediction, labels=labels))   ### Loss function / Cost Function


	optimizer = tf.train.AdamOptimizer().minimize(loss)   ### Optimizer to Optimize the Loss
	epochs = 20   ### Number of Iterations  (Feed Forward with Back Propogation)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			loss_per_epoch = 0   ### container for our updated Loss/Cost

			for x in range(int(data.train.num_examples/batch_size)):   ### Iterate through each Training Example
				features_epochs , labels_epochs = data.train.next_batch(batch_size)

				optimize, cost = sess.run([optimizer, loss], feed_dict={features: features_epochs, labels:labels_epochs})   ### feeding the values of Features and Labels
				loss_per_epoch = loss_per_epoch + cost

			print('Epochs', epoch, 'Loss', loss_per_epoch)

		compare = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)) 
		
		accuracy = tf.reduce_mean(tf.cast(compare, 'float'))

		print('Accuracy', accuracy.eval({features: data.test.images, labels: data.test.labels}))


def read_data():
	mnist = input_data.read_data_sets('MNiST_data', one_hot=True)  ### Read MNIST Data

	return mnist

def main():
	### Number of Neurons(Nodes) per Hidden Layer

	node_hidden_layer_1 = 500
	node_hidden_layer_2 = 500
	node_hidden_layer_3 = 500

	classes = 10   ### Total classes (0-9)
	batch_size = 50  ### Chunk Size(batch Processing)

	### Matric of Height and width
	features = tf.placeholder('float',[None, 784])  ### Input Data
	labels = tf.placeholder('float', )  ### Actual Output

	data = read_data()   ### Call for Data Read Funtion
	train_model(data,features,labels, node_hidden_layer_1, node_hidden_layer_2, node_hidden_layer_3, classes, batch_size)


if __name__ == '__main__':  ### Starting Point of the Program
	main()
