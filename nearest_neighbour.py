from __future__ import division
import collections
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

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

train_images = np.zeros((600,784),order='F')
train_labels = np.zeros((600,1),order='F')
test_images = np.zeros((150,784),order='F')
test_labels = np.zeros((150,1),order='F')

index = 0
p_1 = 0
p_2 = 0
p_7 = 0

for i in range(0, len(labels)):
	
	if labels[i] == 1 and p_1 < 200:
		train_images[index] = images[i]
                train_labels[index] = labels[i]
                p_1 = p_1+1
		index = index+1

	elif labels[i] == 2 and p_2 < 200:
		train_images[index] = images[i]
                train_labels[index] = labels[i]
                p_2 = p_2+1 
		index = index+1
	elif labels[i] == 7 and p_7 < 200:
		train_images[index] = images[i]
                train_labels[index] = labels[i]
                p_7 = p_7+1
		index = index+1
	if (index == 600):
		break;

index = 0
p_1 = 0
p_2 = 0
p_7 = 0

for i in range(len(labels)-1000,0,-1):

        if labels[i] == 1 and p_1 < 50:
                test_images[index] = images[i]
                test_labels[index] = labels[i]
                p_1 = p_1+1
                index = index+1

        elif labels[i] == 2 and p_2 < 50:
                test_images[index] = images[i]
                test_labels[index] = labels[i]
                p_2 = p_2+1
                index = index+1

        elif labels[i] == 7 and p_7 < 50:
                test_images[index] = images[i]
                test_labels[index] = labels[i]
                p_7 = p_7+1
                index = index+1
	if (index == 150):
		break;

print test_images.shape
print test_labels.shape

Accuracy=np.zeros((5,1),order='F')
Average=np.zeros((5,1),order='F')
list = [1,3,5,7,9]
allot = 0
predict=np.zeros((120,1),order='F')
m = 0

for k in list:
	for fold in range(0,5):
		k_test_images = train_images[allot:allot+120]
		k_test_labels = train_labels[allot:allot+120]
		k_train_images = np.append(train_images[:allot],train_images[allot+120:],0 )
		k_train_labels = np.append(train_labels[:allot],train_labels[allot+120:],0 )
		allot =allot+120
		for i in range(0,len(k_test_images)):
			NN = np.zeros((k,1),order='F')
			NNL = np.zeros((k,1),order='F')
			for j in range(0,len(k_train_images)):
				dist = np.sum(np.sqrt(np.square(k_test_images[i]-k_train_images[j])))
				for l in range(0,k):
					if NN[l] == 0: 
						NN[l] = dist
						NNL[l]= k_train_labels[j]
						break
					elif NN[l] > dist:
						NN[l] = dist	
						NNL[l] = k_train_labels[j]	
						break
			
			vals,counts=mode(NNL, axis=0)
			value=np.argmax(counts)
			predict[i]=vals[value]
		C = (predict == k_test_labels)
		Accuracy[fold] = np.sum(C)/len(k_test_labels)
	Average[m] = np.mean(Accuracy)
	m = m+1
	Accuracy=np.zeros((5,1),order='F')
	predict=np.zeros((120,1),order='F')
	allot = 0


Accuracy=np.zeros((5,1),order='F')
Average=np.zeros((5,1),order='F')
list = [1,3,5,7,9]
allot = 0
predict=np.zeros((120,1),order='F')

predict=np.zeros((len(test_labels),1),order='F')
for i in range(0,len(test_images)):
	NN_dist = 0
	NN_label=0
	for j in range(0,len(train_images)):
		dist = np.sum(np.sqrt(np.square(test_images[i]-train_images[j])))
		if NN_dist == 0:
			NN_dist = dist
			NN_label= train_labels[j]
		elif NN_dist > dist:
			NN_dist = dist
			NN_label = train_labels[j]
	predict[i]=NN_label
shuffle = []
shuffle = np.append(predict,test_labels,1)
print shuffle
C = (predict == test_labels)
Accuracy_final = np.sum(C)/len(test_labels)
print Accuracy_final
