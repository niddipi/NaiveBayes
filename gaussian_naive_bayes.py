from __future__ import division
import collections
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as norm

images_file = open('train-images.idx3-ubyte' , 'rb')
images = images_file.read()
images_file.close()
images = bytearray(images)
images = images[16:]
images = np.array(images)
images = images.reshape(60000,784)

labels_file = open('train-labels.idx1-ubyte' , 'rb')
labels = labels_file.read()
labels_file.close()

labels = bytearray(labels)
labels = labels[8:]
labels = np.array(labels)
labels=labels.reshape(60000,1)
shuffle =[]

dataset_images = np.zeros((2000,784),order='F')
dataset_labels = np.zeros((2000,1),order='F')

index = 0
p_0 = 0
p_1 = 0
p_2 = 0
p_3 = 0
p_4 = 0
p_5 = 0
p_6 = 0
p_7 = 0
p_8 = 0
p_9 = 0

for i in range(0, len(labels)):
	
	if labels[i] == 5 and p_5 < 1000:
		dataset_images[index] = images[i]
		dataset_labels[index] = labels[i]
		p_5=p_5+1
		index = index+1
	elif labels[i] == 0 and p_0 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_0 = p_0+1 
		index = index+1
		
	elif labels[i] == 1 and p_1 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_1 = p_1+1
		index = index+1
	elif labels[i] == 2 and p_2 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_2 = p_2+1 
		index = index+1
	elif labels[i] == 3 and p_3 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_3 = p_3+1
		index = index+1
	elif labels[i] == 4 and p_4 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_4 = p_4+1 
		index = index+1
	elif labels[i] == 6 and p_6 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_6 = p_6+1
		index = index+1
	elif labels[i] == 7 and p_7 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_7 = p_7+1
		index = index+1
	elif labels[i] == 8 and p_8 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_8 = p_8+1
		index = index+1
	elif labels[i] == 9 and p_9 < 111:
		dataset_images[index] = images[i]
                dataset_labels[index] = labels[i]
                p_9 = p_9+1
		index = index+1
	if (index == 2000):
		break;

shuffle = np.append(dataset_images,dataset_labels,1)
shuffle  = np.array(shuffle)
np.random.shuffle(shuffle)
print shuffle.shape 
dataset_images = shuffle[:,0:784]
dataset_labels = shuffle[:,[784]]
#print dataset_images.shape
#print dataset_labels.shape

train_images = dataset_images[200:2000]
train_labels = dataset_labels[200:2000]

test_images  = dataset_images[0:200]
test_labels  = dataset_labels[0:200]

likelihood = np.zeros((10,784),order='F')
prior_class = np.zeros((10,1),order='F')
prior_class_prob = np.zeros((10,1),order='F')
mean = np.zeros((10,784),order='F')
var = np.zeros((10,784),order='F')
predict_case_1 = np.zeros((10,1),order='F') 
predict_case_2 = np.zeros((10,1),order='F') 
predict_case_3 = np.zeros((10,1),order='F') 
predict_case_4 = np.zeros((10,1),order='F') 
predict_case_5 = np.zeros((10,1),order='F') 
index = np.zeros((len(test_images),1),order='F')

'''
train_labels=train_labels.astype(np.int64)

for i in range(0,len(train_images)):
        k = train_labels[i]
        likelihood[k] = likelihood[k] + train_images[i]
	var[k] = var[k] + ((train_images[i] - mean[k])**2)
        prior_class[k] = prior_class[k]+1

for i in range(0,10):
	mean[i] = np.divide(likelihood[i],prior_class[i])
	var[i] = var[i] / prior_class[i]

'''
prior_class = np.array([(train_labels == i).sum() for i in range(0,len(train_images))], dtype=np.float)
for i in range(0,10):
	sum = np.sum(train_images[n] if train_labels[n] == i else 0.0 for n in range(0,len(train_images)))
	mean[i] = sum / prior_class[i]
for i in range(0,10):
	sum = np.sum(np.square(train_images[n] - mean[i]) if train_labels[n] == i else 0.0 for n in range(0,len(train_images)))
	var[i] = sum / prior_class[i] 

prior_class_prob = prior_class/len(train_images)
test_prob = []
test_prob1 = []
test_prob2 = []

value = math.sqrt(6.28)
for i in range(0,len(test_images)):
	for each_class in range(0,10):
		test_prob = np.divide(((np.square(test_images[i]-mean[each_class]))),(2*(var[each_class])+1))
		test_prob = np.sum(test_prob)
		test_prob1 = value * np.sqrt(var[each_class])
		test_prob1 = np.nan_to_num(test_prob1)
		test_prob1[test_prob1 == 0] = 1
		test_prob2 = 2 *((var[each_class]))
		test_prob2[test_prob2 == 0] = 1
		
		predict[each_class] = (test_prob) *( -np.log(np.sum(test_prob1)) + np.log(prior_class_prob[each_class]))	
	index[i]=np.argmax(predict)

index[index != 5] = 0
index[index == 5] = 1
test_labels[test_labels != 5] = 0
test_labels[test_labels ==5] = 1

C = (index == test_labels)
print "Accuracy is :"
print np.sum(C)/float(len(C))
#test_prob[test_prob == 0] = 1
shuffle = np.append(index,test_labels,1)
#print shuffle
TP = np.sum(np.logical_and(index == 1, test_labels == 1))
TN = np.sum(np.logical_and(index == 0, test_labels == 0))
FP = np.sum(np.logical_and(index == 1, test_labels == 0))
FN = np.sum(np.logical_and(index == 0, test_labels == 1))
TPR = (TP/(TP+FN))
FPR = (FP/(TN+FP))

plt.plot(FPR,TPR)
plt.show() 
