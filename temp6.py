import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn.metrics as met
from xgboost import XGBClassifier

from mlxtend.data import loadlocal_mnist
import platform

def addNoisePatch(Xarray,d1,d2,sd1,sd2,scale=255):
    rand = np.random.rand(Xarray.shape[0],2)
    for index in range(0,Xarray.shape[0]):
        islice = Xarray[index,:]
        Xim = np.reshape(islice,(d1,d2))
        upperBound1 = d1-sd1+0.99
        upperBound2 = d2-sd2+0.99
        shift1 = int(upperBound1*rand[index,0])
        shift2 = int(upperBound2*rand[index,1])
        Xim[shift1:shift1+sd1,shift2:shift2+sd2] = scale*np.random.rand(sd1,sd2)
        Xarray[index,:] = np.reshape(Xim,islice.shape)
    return Xarray



plt.close("all")


dir = r"C:\Users\night\Downloads\MNIST"
scaleFactor = 0.3
#wx = 14
wxStep = 5
wy = 15
#noisyTraining = False

trainingSampleSize = 12000
testSampleSize = 4200
valCount = 5

storedModels = ["",""] * valCount

if not platform.system() == 'Windows':
    p_train_X, p_train_y = loadlocal_mnist(
            images_path=dir+"\\"+'train-images-idx3-ubyte', 
            labels_path=dir+"\\"+'train-labels-idx1-ubyte')
    p_test_X, p_test_y = loadlocal_mnist(
            images_path=dir+"\\"+'t10k-images-idx3-ubyte', 
            labels_path=dir+"\\"+'t10k-labels-idx1-ubyte')

else:
    p_train_X, p_train_y = loadlocal_mnist(
            images_path=dir+"\\"+'train-images.idx3-ubyte', 
            labels_path=dir+"\\"+'train-labels.idx1-ubyte')
    p_test_X, p_test_y = loadlocal_mnist(
            images_path=dir+"\\"+'t10k-images.idx3-ubyte', 
            labels_path=dir+"\\"+'t10k-labels.idx1-ubyte')

masterResultsList_noisyTraining = list()
masterResultsList_no_noisyTraining = list()

for run in range(0,2*valCount):
    if run%2 == 0:
        noisyTraining = False
    else:
        noisyTraining = True
    
    wx = wxStep *int(run/2)   
    ind = np.arange(0,p_train_y.shape[0])
    np.random.shuffle(ind)    
    np_train_X = np.int_(scaleFactor*p_train_X[ind,:])
    np_train_y = p_train_y[ind]
    
    ind_test = np.arange(0,p_test_y.shape[0])
    np.random.shuffle(ind_test)
    np_test_X = np.int_(scaleFactor*p_test_X[ind_test,:])
    np_test_y = p_test_y[ind_test]
    
    
    if noisyTraining:
        train_X = addNoisePatch(np_train_X[0:trainingSampleSize,:],28,28,wx,wy)
    else: 
        train_X = np_train_X[0:trainingSampleSize,:]
    
    train_y = np_train_y[0:trainingSampleSize]
    
    #test_X = np_test_X[0:testSampleSize,:]
    test_X = addNoisePatch(np_test_X[0:testSampleSize,:],28,28,wx,wy)
    
    test_y = np_test_y[0:testSampleSize]
    
    storedModels[run] = XGBClassifier(booster = 'gbtree', max_depth=9,n_estimators = 300)
    storedModels[run].fit(train_X, train_y)
    #model = XGBClassifier()
    #model.fit(train_X, train_y)
    
    # Simulate the predictions in the clear
    #y_pred_clear = model.predict(test_X)
    y_pred_clear = storedModels[run].predict(test_X)
    
    accuracy_score = met.accuracy_score(test_y,y_pred_clear)
    
    res = (wx, noisyTraining, accuracy_score)
    print(res)
    if noisyTraining:
        masterResultsList_noisyTraining.append(res)
    else:
        masterResultsList_no_noisyTraining.append(res)
    # noisyTrain_X = addNoisePatch(train_X, 28, 28, 6, 3)
    if run == 0:
        for q in range(80,85):
            plt.figure(q)
            plt.imshow(np.reshape(test_X[q,:],(28,28)))
            
        for q in range(90,95):
            plt.figure(q)
            plt.imshow(np.reshape(train_X[q,:],(28,28)))

# %%
plt.close("all")

l = valCount
stats = np.zeros((2,2,l))
normalization = wy/(28*28)
for pt in range(0,l):
    stats[0,0,pt] = masterResultsList_no_noisyTraining[pt][0]
    stats[0,1,pt] = masterResultsList_no_noisyTraining[pt][2]
    stats[1,0,pt] = masterResultsList_noisyTraining[pt][0]
    stats[1,1,pt] = masterResultsList_noisyTraining[pt][2]
plt.figure(470)
plt.plot(normalization*stats[0,0,:],stats[0,1,:],'-*',label = "No-noise training")
plt.plot(normalization*stats[1,0,:],stats[1,1,:],'-*',label = "Noisy training")
plt.legend()
plt.xlabel("Proportion covered by noisy block")
plt.ylabel("Accuracy score")
plt.ylim([0,1])
plt.title(("Accuracy of noisy MNIST" + "\n" + "Signal factor = "+str(scaleFactor)+"\n"))