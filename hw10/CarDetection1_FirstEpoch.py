
# coding: utf-8

# In[6]:


# Usage: python CarDetection1_FirstEpoch.py 10 5 200 10
#                    S, T, StepOfCls, StepOfTh
# V2 changes: print (TP, ...) at the end of each weak classifier selection

import cv2 as cv
import numpy as np
import time
import copy
import sys
TrainImages = []
for x in range(1,710+1):
    TrainImages.append('ECE661_2018_hw10_DB2/train/positive/%06d.png' %x)
for x in range(1,1758+1):
    TrainImages.append('ECE661_2018_hw10_DB2/train/negative/%06d.png' %x)
TestImages = []
for x in range(711,888+1):
    TestImages.append('ECE661_2018_hw10_DB2/test/positive/%06d.png' %x)
for x in range(1759,2198+1):
    TestImages.append('ECE661_2018_hw10_DB2/test/negative/%06d.png' %x)


# In[7]:

def GetResult(Pred_Final, TestGT):
    TP = np.sum(np.logical_and(Pred_Final==1, TestGT==1))
    FP = np.sum(np.logical_and(Pred_Final==1, TestGT==0))
    TN = np.sum(np.logical_and(Pred_Final==0, TestGT==0))
    FN = np.sum(np.logical_and(Pred_Final==0, TestGT==1))
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return TP, FP, TN, FN, TPR, FPR


def Haar(im): # Given (20,40) img, return (166000,) long Haar feature values
    Integral = np.zeros((im.shape[0]+1, im.shape[1]+1))
    # Compute Integral
    for H in range(1, im.shape[0]+1):
        LineSum = 0
        for W in range(1, im.shape[0]+1):
            LineSum = LineSum + im[H-1, W-1]
            Integral[H, W] = Integral[H-1, W] + LineSum

    HaarResults, ClassifierCnt = (np.zeros(166000, ), 0)
    # Horizontal Haar (1~20,1~20x2), pattern [-1, 1]
    for H in range(1,21):
        for W in range(1,21):
            # Sliding window
            for H0 in range(21-H):
                for W0 in range(41-2*W):              
                    HaarResults[ClassifierCnt] = -Integral[H0,W0] + Integral[H0+H,W0]                                                  +2*Integral[H0,W0+W] - 2*Integral[H0+H,W0+W]                                                  -Integral[H0,W0+W*2] + Integral[H0+H,W0+W*2]
                    ClassifierCnt = ClassifierCnt + 1
    # Vertical Haar (1~10x2,1~40), pattern [-1, 1]^T
    for H in range(1,11):
        for W in range(1,41):
            # Sliding window
            for H0 in range(21-2*H):
                for W0 in range(41-W):
                    HaarResults[ClassifierCnt] = -Integral[H0,W0] + Integral[H0,W0+W]                                                  +2*Integral[H0+H,W0] - 2*Integral[H0+H,W0+W]                                                  -Integral[H0+2*H,W0] + Integral[H0+2*H,W0+W]
                    ClassifierCnt = ClassifierCnt + 1   
    return HaarResults



# In[22]:


# HaarResults  166k x 2468
HaarResults = np.load("HaarResults.npy")
TestHaarResults = np.load("TestHaarResults.npy")
GT = np.array([1]*710 + [0]*1758)
TestGT = np.array([1]*178 + [0]*440)
InitWt = np.array([1/710/2]*710 + [1/1758/2]*1758)

# HaarResults  166k x 2469, uses the threshold "<"
#print("Max of HaarResults = ", np.max(HaarResults))
HaarResults = np.hstack([HaarResults, np.ones((166000,1))*1e7])


# In[27]:


import pickle
S, T, StepOfCls, StepOfTh = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
Wt = np.zeros((T+1, 2468))
Wt[0, :] = np.copy(InitWt)
alpha, beta, SltWeakCls, SltThIdx = (np.zeros((T,)), np.zeros((T,)), 
                                     np.zeros((T,)).astype("int"), np.zeros((T,)).astype("int"))
time1 = time.time()
for t in range(T):  # T iteration of picking weak classifier
    print("Iteration #%d of picking weak classifier" %t)
    BestSpecInCls = [(1e7, -1)]*HaarResults.shape[0]
    for ClsIdx in range(HaarResults.shape[0]):
        if ClsIdx%StepOfCls != 0:
            BestSpecInCls[ClsIdx] = (1e7, -1)  # Error is large enough
            continue
        # Choose the best Th. (2468+1) and polarity (+,-), save in BestSpecInCls[ClsIdx]
        MinWtErr, MinThIdx = (1., -1)
        # Positive (+) first: < means "negative"(0), >= means "positive" 1
        for ThIdx, Th in enumerate(list(HaarResults[ClsIdx, :])):
            if ThIdx%StepOfTh != 0:
                continue
            Pred = (HaarResults[ClsIdx, 0:-1]>=Th)
            Wrong = np.not_equal(Pred, GT)
            WtErr = np.sum(Wt[t, :]*Wrong)
            if(WtErr < MinWtErr):
                MinWtErr, MinThIdx = (WtErr, ThIdx)
        # Negative (-) next: >= means "negative"(0), < means "positive" 1        
        for ThIdx, Th in enumerate(list(HaarResults[ClsIdx, :])):
            if ThIdx%StepOfTh != 0:
                continue
            Pred = (HaarResults[ClsIdx, 0:-1]<Th)
            Wrong = np.not_equal(Pred, GT)
            WtErr = np.sum(Wt[t, :]*Wrong)
            if(WtErr < MinWtErr):
                MinWtErr, MinThIdx = (WtErr, ThIdx+HaarResults.shape[1])
        if(ClsIdx%40000==0):      
            if(MinThIdx<HaarResults.shape[1]): # Positive polarity
                print("ClsIdx = %-3d, MinWtErr, MinThIdx, MinTh = (%.4f, %-4d, %3.0f) (+)" 
                      %(ClsIdx, MinWtErr, MinThIdx, HaarResults[ClsIdx, MinThIdx]))
            else:   
                print("ClsIdx = %-3d, MinWtErr, MinThIdx, MinTh = (%.4f, %-4d, %3.0f) (-)" 
                      %(ClsIdx, MinWtErr, MinThIdx, HaarResults[ClsIdx, MinThIdx-HaarResults.shape[1]]))
        BestSpecInCls[ClsIdx] = (MinWtErr, MinThIdx)
    np.save("save/BestSpecInCls_it%d.npy" %(t), BestSpecInCls)
    
    # Select the best weak weak classifier
    MinWtErrs = [x[0] for x in BestSpecInCls]
    MinErr = np.min(MinWtErrs)
    SltWeakCls[t] = int(np.argmin(MinWtErrs))    
    SltThIdx[t] = BestSpecInCls[SltWeakCls[t]][1]
    Th = HaarResults[SltWeakCls[t], int(SltThIdx[t])%HaarResults.shape[1]] 
    print("Weak classifier #%d is selected, MinWtErr, MinThIdx, MinTh = (%.4f, %-4d, %3.0f)" 
          %(SltWeakCls[t], BestSpecInCls[SltWeakCls[t]][0], SltThIdx[t], Th))
    # Update weight
    beta[t] = MinErr/(1-MinErr)
    alpha[t] = np.log(1/beta[t])
    if(SltThIdx[t] < HaarResults.shape[1]): # Positive polarity
        Pred = (HaarResults[SltWeakCls[t], 0:-1]>=Th)
        Wrong = np.not_equal(Pred, GT)
        # Wrong = 1 then +=0, Wrong = 0, then +=(beta-1)*Wt
        Wt[t+1, :] = Wt[t, :] + (beta[t] - 1) * Wt[t, :] * (1-Wrong)
    else: # Negative polarity
        Pred = (HaarResults[SltWeakCls[t], 0:-1]<Th)
        Wrong = np.not_equal(Pred, GT)
        # Wrong = 1 then +=0, Wrong = 0, then +=(beta-1)*Wt
        Wt[t+1, :] = Wt[t, :] + (beta[t] - 1) * Wt[t, :] * (1-Wrong)
    TP, FP, TN, FN, TPR, FPR = GetResult(Pred, GT)
    print("TP, FP, TN, FN, TPR, FPR = (%d, %d, %d, %d, %.4f, %.4f)" %(TP, FP, TN, FN, TPR, FPR))

    Wt[t+1, :] = Wt[t+1, :] / np.sum(Wt[t+1, :])
    
    # Saving the varibles
    SaveItem = [SltWeakCls[t], SltThIdx[t], 
                beta[t], Pred, Wrong, 
                Wt[t, :], Wt[t+1, :]]
    with open("save/Debugging_it%d.npy" %t, 'wb') as f:
        pickle.dump(SaveItem, f)
time2 = time.time()
print("Training time: %.2f s" %(time2 - time1))


# In[11]:




# In[24]:


# Summarization printing
fout = open("Summarization.txt", "w")
print("The %d weak classifies are as follows:" %T)
print("The %d weak classifies are as follows:" %T, file = fout)
print("%-12s %-12s %-10s" %("SltWeakCls", "SltThIdx", "alpha"))
print("%-12s %-12s %-10s" %("SltWeakCls", "SltThIdx", "alpha"), file = fout)
for t in range(T):
    print("%-12d %-12d %-10.4f" %(SltWeakCls[t], SltThIdx[t], alpha[t]))
    print("%-12d %-12d %-10.4f" %(SltWeakCls[t], SltThIdx[t], alpha[t]), file = fout)
fout.close()
# Final strong classifier: Use alpha, SltWeakCls, SltThIdx to construct
#    Eq: np.sum(alpha * h(x)) - 0.5*sum(alpha) >= 0
# Compute final pred, (2468,) shape 0-1 vector
# Compute false-postive rate->FP/(FP+TN) should be 30% 
# Compute true-postive rate->TP/(TP+FN) should be 99% or 1
print("Testing on training set")
Feature_Final = np.zeros(GT.shape)
Dec_Th = np.sum(alpha)/2
for t in range(T):
    # In this iteration SltWeakCls[t] is selected, with Th @ SltThIdx[t]
    Feature = HaarResults[SltWeakCls[t], :-1]
    Th = HaarResults[SltWeakCls[t], int(SltThIdx[t])%HaarResults.shape[1]] 
    if(SltThIdx[t] < HaarResults.shape[1]): # Positive polarity
        Pred = (Feature>=Th)
        Feature_Final = Feature_Final + Pred * alpha[t]
    else: # Negative polarity
        Pred = (Feature<Th)
        Feature_Final = Feature_Final + Pred * alpha[t]
Pred_Final = (Feature_Final > Dec_Th)
TP, FP, TN, FN, TPR, FPR = GetResult(Pred_Final, GT)
print("TP, FP, TN, FN, TPR, FPR = (%d, %d, %d, %d, %.4f, %.4f)" %(TP, FP, TN, FN, TPR, FPR))


# In[25]:


# Also check in test test
Feature_Final = np.zeros(TestGT.shape)
Dec_Th = np.sum(alpha)/2
print("Testing on test set")
for t in range(T):
    # In this iteration SltWeakCls[t] is selected, with Th @ SltThIdx[t]
    Feature = TestHaarResults[SltWeakCls[t], :]
    Th = HaarResults[SltWeakCls[t], int(SltThIdx[t])%HaarResults.shape[1]] 
    if(SltThIdx[t] < HaarResults.shape[1]): # Positive polarity
        Pred = (Feature>=Th)
        Feature_Final = Feature_Final + Pred * alpha[t]
    else: # Negative polarity
        Pred = (Feature<Th)
        Feature_Final = Feature_Final + Pred * alpha[t]
Pred_Final = (Feature_Final > Dec_Th)
TP, FP, TN, FN, TPR, FPR = GetResult(Pred_Final, TestGT)
print("TP, FP, TN, FN, TPR, FPR = (%d, %d, %d, %d, %.4f, %.4f)" %(TP, FP, TN, FN, TPR, FPR))


