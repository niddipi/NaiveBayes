import collections
import numpy as np
import matplotlib.pyplot as plt

training_images_file = open('train-images.idx3-ubyte' , 'rb')
training_images = training_images_file.read()
training_images_file.close()

training_images = bytearray(training_images)

training_images = training_images[16:]
training_images = np.array(training_images)
training_images = training_images.reshape(60000,784)
training_images = np.multiply(training_images, 1.0 / 255.0)
image1 = training_images[52].reshape([28,28])
training_labels_file = open('train-labels.idx1-ubyte' , 'rb')
training_labels = training_labels_file.read()
training_labels_file.close()

training_labels = bytearray(training_labels)
training_labels = training_labels[8:]
training_labels = np.array(training_labels)
training_labels=training_labels.reshape(60000,1)

train_images = []
probability_each_pixel = []
likelihood = np.zeros((10,784),order='F')
likelihood_total_each_pixel = np.zeros((1,784),order='F')
prior_class = np.zeros((10,1),order='F')
train_images = training_images
train_images = np.array(train_images)
train_images[train_images>=0.5] = 1
train_images[train_images<0.5] = 0

test_images_file = open('t10k-images.idx3-ubyte' , 'rb')
test_images = test_images_file.read()
test_images_file.close()

test_images = bytearray(test_images)

test_images = test_images[16:]
test_images = np.array(test_images)
test_images = test_images.reshape(10000,784)
test_images = np.multiply(test_images, 1.0 / 255.0)
test_images[test_images>=0.5] = 1
test_images[test_images<0.5] = 0
image1 = test_images[52].reshape([28,28])
test_labels_file = open('t10k-labels.idx1-ubyte' , 'rb')
test_labels = test_labels_file.read()
test_labels_file.close()

test_labels = bytearray(test_labels)
test_labels = np.array(test_labels)

test_labels = test_labels[8:]
test_labels = np.array(test_labels)
test_labels=test_labels.reshape(10000,1)
for i in range(0,60000):
	k = training_labels[i]
	likelihood[k] = likelihood[k] + train_images[i]
	prior_class[k] = prior_class[k]+1

for i in range(0,10):
	likelihood_total_each_pixel = likelihood_total_each_pixel + likelihood[i]

for i in range(0,10):
	likelihood[i] =  (likelihood[i]+1)/(likelihood_total_each_pixel+1)
#	likelihood[i] =  (likelihood[i]+1)/(prior_class[i])
'''
likelihood_total_each_pixel = np.sum(likelihood[n] for n in range(0,len(likelihood)))


prior_class = np.array([(training_labels == i).sum() for i in range(0,10)], dtype=np.float)
prior_class = prior_class/len(train_images)
for i in range(0,10):
        sum = np.sum(train_images[n] if training_labels[n] == i else 0.0 for n in range(0,len(train_images)))
        likelihood[i] = sum

'''

#likelihood_total_each_pixel[likelihood_total_each_pixel == 0] = 1
#probability_each_pixel = (likelihood+1)/(likelihood_total_each_pixel+1)
#probability_each_pixel = likelihood/prior_class
prior_class = prior_class/len(train_images)

output = np.zeros((10,1),order='F')
test_prob = []
prior_prob = []
index = np.zeros((len(test_images),1),order='F')
prior_prob = np.log(prior_class)

for p in range(0,len(test_images)):
	for i in range(0,10):
#		test_prob = (np.multiply(test_images[p],probability_each_pixel[i]))
		test_prob = (np.multiply(test_images[p],likelihood[i]))
		#test_prob = (np.multiply(prior_class[i],test_prob))
		test_prob[test_prob == 0] = 1
		test_prob = np.log(test_prob)
		output[i] = np.log(prior_class[i]) + np.sum(test_prob)
	index[p] = np.argmax(output)
print index
C = (index == test_labels)
print "Accuracy is :"
print np.sum(C)/float(len(C))
'''
for i in range(0,10):
	test_prob = (np.multiply(test_images[212],probability_each_pixel[i]))
#	test_prob = np.nan_to_num(test_prob)
	test_prob[test_prob == 0] = 1
	test_prob = np.log(test_prob)
	test_prob = np.array(test_prob)
	output[i] = np.log(prior_class[i]) + np.sum(test_prob)
ind = np.argmax(output)

print ind
print test_labels[219]
image= test_images[219].reshape([28,28])
f = plt.figure(1)
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
f.show()
plt.show()
'''

