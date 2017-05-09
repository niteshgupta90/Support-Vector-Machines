import sys
import math
import re
import random as rn
import numpy as np
from sklearn import svm
from sklearn import cross_validation
import scipy.optimize as sci

trainData = "features.train.txt"
testData  = "features.test.txt"

def readInput(file=trainData):
  data = np.genfromtxt(file, dtype=float)
  return(np.apply_along_axis(lambda(x): x[1:3],1,data), np.apply_along_axis(lambda(x): x[0],1,data))

def getBinaryClassification(y,choice=0):
  classifiedData=np.ones(len(y))
  classifiedData[y!=choice] = -1
  return(classifiedData)

def svmRun(x,y, C=0.01, Q=2):
  clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1, coef0=1)
  clf.fit(x, y) 
  yhat = clf.predict(x)
  Etrain = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Etrain':Etrain,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def question2_2_1():
  (x,y) = readInput(trainData)
  class0 = svmRun(x,getBinaryClassification(y,choice=0), C=0.01, Q=2)
  class2 = svmRun(x,getBinaryClassification(y,choice=2), C=0.01, Q=2)
  class4 = svmRun(x,getBinaryClassification(y,choice=4), C=0.01, Q=2)
  class6 = svmRun(x,getBinaryClassification(y,choice=6), C=0.01, Q=2)
  class8 = svmRun(x,getBinaryClassification(y,choice=8), C=0.01, Q=2)
  return({
    'Etrain_0vsALL':class0['Etrain'],
    'Etrain_2vsALL':class2['Etrain'],
    'Etrain_4vsALL':class4['Etrain'],
    'Etrain_6vsALL':class6['Etrain'],
    'Etrain_8vsALL':class8['Etrain'], })

def question2_2_2():
  (x,y) = readInput(trainData)
  class1 = svmRun(x,getBinaryClassification(y, choice=1), C=0.01, Q=2)
  class3 = svmRun(x,getBinaryClassification(y, choice=3), C=0.01, Q=2)
  class5 = svmRun(x,getBinaryClassification(y, choice=5), C=0.01, Q=2)
  class7 = svmRun(x,getBinaryClassification(y, choice=7), C=0.01, Q=2)
  class9 = svmRun(x,getBinaryClassification(y, choice=9), C=0.01, Q=2)
  return({
         'Etrain_1vsALL':class1['Etrain'],
         'Etrain_3vsALL':class3['Etrain'],
         'Etrain_5vsALL':class5['Etrain'],
         'Etrain_7vsALL':class7['Etrain'],
         'Etrain_9vsALL':class9['Etrain'], })

def question2_2_3():
  (x,y) = readInput(trainData)
  return(svmRun(x,getBinaryClassification(y,choice=0))['n_support'] -
         svmRun(x,getBinaryClassification(y,choice=1))['n_support'])

def question2_2_4(C_vector=[0.001,0.01,0.1,1.0], Q=2):
  (x,y)       = readInput(trainData)
  idx = np.logical_or(y==1,y==5)
  x   = x[idx,:]
  y   = getBinaryClassification(y[idx],1)
  (xTest,yTest) = readInput(testData)
  idx  = np.logical_or(yTest==1,yTest==5)
  xTest = xTest[idx,:]
  yTest = getBinaryClassification(yTest[idx],1)
  Etrain  = []
  Etest = []
  numOfSV  = []
  for C in C_vector:
    r = svmRun(x,y, C=C, Q=Q)
    Etrain.append(r['Etrain'])
    numOfSV.append(r['n_support'])
    clf  = r['clf']
    yhat = clf.predict(xTest)
    Etest.append( np.sum( yTest*yhat < 0 ) / (1.*yTest.size) )
  return({'Etrain':Etrain,'Etest':Etest,'numOfSV':numOfSV})

def question2_2_5():
  r2 = question2_2_4(C_vector=[0.0001,0.001,0.01,0.1,1.0], Q=2)
  r5 = question2_2_4(C_vector=[0.0001,0.001,0.01,0.1,1.0], Q=5)
  return({'Q2':r2, 'Q5':r5})

# k-fold cross validation version of SVM
def svmRun_cv(x,y,C=0.0001,Q=2,folds=10):
  kFoldCV = cross_validation.KFold(len(y), n_folds=folds, shuffle=True)
  Ecv = np.array([])
  Etrain = np.array([])
  numOfSV = np.array([])
  i = 0
  for train,test in kFoldCV:
    x_train, x_test, y_train, y_test = x[train],x[test],y[train],y[test]
    # print('fold %d: train_n %d, test_n %d' % (i, len(train), len(test)))
    r = svmRun(x_train,y_train,C=C,Q=Q)
    Etrain = np.append(Etrain, r['Etrain'])
    numOfSV = np.append(numOfSV, r['n_support'])
    clf  = r['clf']
    y_pred = clf.predict(x_test) 
    Ecv = np.append(Ecv, np.sum(y_test*y_pred<0) / (1.*y_pred.size) )
    i += 1
  return({'Ecv':np.mean(Ecv), 'Etrain':np.mean(Etrain), 'numOfSV':np.mean(numOfSV)})

def question_2_3_1(C_vector=[0.0001,0.001,0.01,0.1,1.0], runs=100):
  (x,y) = readInput(trainData)
  idx   = np.logical_or(y==1,y==5)
  x     = x[idx,:]
  y     = getBinaryClassification(y[idx], choice=1)
  numOfWins  = [0 for i in range(len(C_vector))]
  Ecv   = np.empty( (runs,len(C_vector)) )
  for i in range(runs):
    for j in range(len(C_vector)):
      C = C_vector[j]
      r = svmRun_cv(x,y,C=C,Q=2,folds=10)
      Ecv[i,j] = r['Ecv']
      # Ecv = np.append(Ecv, r['Ecv'])
    idx = np.argmin(Ecv[i,:])
    numOfWins[idx] += 1
  return({'numOfWins':zip(C_vector,numOfWins), 'Ecv': zip(C_vector,np.mean(Ecv,1).tolist())})

def question_2_3_2(C=0.01, runs=100):
  (x,y) = readInput(trainData)
  idx   = np.logical_or(y==1,y==5)
  x     = x[idx,:]
  y     = getBinaryClassification(y[idx], choice=1)
  (xTest,yTest) = readInput(testData)
  idx  = np.logical_or(yTest==1,yTest==5)
  xTest = xTest[idx,:]
  yTest = getBinaryClassification(yTest[idx],1)
  Etrain = []
  Etest = []
  r = svmRun(x,y,C=C,Q=2)
  Etrain.append(r['Etrain'])
  clf  = r['clf']
  yhat = clf.predict(xTest)
  Etest.append( np.sum( yTest*yhat < 0 ) / (1.*yTest.size) )
  return({'Etrain':Etrain,'Etest':Etest})

def svmRun_Gaussian(x,y, C=0.01):
  clf = svm.SVC(kernel='rbf', C=C, gamma=1.)
  clf.fit(x, y) 
  yhat = clf.predict(x)
  Etrain = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Etrain':Etrain, 'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def question_2_4_1and2(C_vector=[0.01,1,100,10**4,10**6]):
  (x,y)       = readInput(trainData)
  idx = np.logical_or(y==1,y==5)
  x   = x[idx,:]
  y   = getBinaryClassification(y[idx],1)
  (xTest,yTest) = readInput(testData)
  idx  = np.logical_or(yTest==1,yTest==5)
  xTest = xTest[idx,:]
  yTest = getBinaryClassification(yTest[idx],1)
  Etrain  = []
  Etest = []
  numOfSV  = []
  for C in C_vector:
    r = svmRun_Gaussian(x,y, C=C)
    Etrain.append(r['Etrain'])
    numOfSV.append(r['n_support'])
    clf  = r['clf']
    yhat = clf.predict(xTest)
    Etest.append( np.sum( yTest*yhat < 0 ) / (1.*yTest.size) )
  return({'Etrain':Etrain,'Etest':Etest,'numOfSV':numOfSV})

x,y = readInput()
svmRun(x,y, C=0.01, Q=2)
svmRun_cv(x,y,C=0.0001,Q=2,folds=10)
svmRun_Gaussian(x,y, C=0.01)
print "\n"

val1 = question2_2_1()
print val1
print "\n"

val2 = question2_2_2()
print val2
print "\n"

val3 = question2_2_3()
print val3
print "\n"

val4 = question2_2_4()
print val4
print "\n"

val5 = question2_2_5()
print val5
print "\n"

val6 = question_2_3_1()
print val6
print "\n"

val7 = question_2_3_2()
print val7
print "\n"

val8 = question_2_4_1and2()
print val8
print "\n"

