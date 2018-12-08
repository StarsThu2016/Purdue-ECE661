
# coding: utf-8

# In[3]:


# Usage: python CarDetection1_NextEpoch.py 9 20 2 2
#                    S, T, StepOfCls, StepOfTh
# Mask_epoch0.npy is a 2468-long 0-1 vector that explicitly tell which images are positive at this stage
import cv2 as cv
import numpy as np
import time
import copy
import sys

def GetResult(Pred_Final, TestGT):
    TP = np.sum(np.logical_and(Pred_Final==1, TestGT==1))
    FP = np.sum(np.logical_and(Pred_Final==1, TestGT==0))
    TN = np.sum(np.logical_and(Pred_Final==0, TestGT==0))
    FN = np.sum(np.logical_and(Pred_Final==0, TestGT==1))
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return TP, FP, TN, FN, TPR, FPR


# In[16]:


import pickle
S, T, StepOfCls, StepOfTh = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))  # 20k speed-up in laptop
Base_Stage = 1

for Stage in range(S):  # It is actually #(Base_Stage+Stage) Stage
    # HaarResults  166k x 2468
    HaarResults = np.load("HaarResults.npy")
    Mask = np.load("Mask_epoch_%d.npy" %(Base_Stage+Stage-1)) 
    HaarResults = HaarResults[:, Mask]
    HaarResults = np.hstack([HaarResults, np.ones((166000,1))*1e7])
    GT = np.array([1]*710 + [0]*1758)
    GT = GT[Mask]
    InitWt = np.array([1/np.sum(GT)/2]*710 + [1/np.sum(GT==0)/2]*1758)
    InitWt = InitWt[Mask]
    print("np.sum(InitWt) = ", np.sum(InitWt))

    TestHaarResults = np.load("TestHaarResults.npy")
    TestGT = np.array([1]*178 + [0]*440)

    Wt = np.zeros((T+1, InitWt.shape[0]))
    Wt[0, :] = np.copy(InitWt)
    alpha, beta, SltWeakCls, SltThIdx = (np.zeros((T,)), np.zeros((T,)), 
                                         np.zeros((T,)).astype("int"), np.zeros((T,)).astype("int"))

    time1 = time.time()
    for t in range(T):  # T iteration of picking weak classifier
        print("Iteration #%d/%d in stage %d/%d of picking weak classifier" 
              %(t, T, Stage+Base_Stage, S+Base_Stage))
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
        np.save("save/BestSpecInCls_it%d_stage%d.npy" %(t,Stage+Base_Stage), BestSpecInCls)

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
        SaveItem = [SltWeakCls[t], SltThIdx[t], beta[t], Pred, Wrong, Wt[t, :], Wt[t+1, :]]
        with open("save/Debugging_it%d_stage%d.npy" %(t,Stage+Base_Stage), 'wb') as f:
            pickle.dump(SaveItem, f)
    time2 = time.time()
    print("Training time per stage: %.2f s" %(time2 - time1))
    
    # Summarization printing
    fout = open("Summarization_Stage%d.txt" %(Base_Stage+Stage), "w")
    print("The %d weak classifiers are as follows (time = %.2f s):" %(T, time2 - time1))
    print("The %d weak classifiers are as follows (time = %.2f s):" %(T, time2 - time1), file = fout)
    print("%-12s %-12s %-10s" %("SltWeakCls", "SltThIdx", "alpha"))
    print("%-12s %-12s %-10s" %("SltWeakCls", "SltThIdx", "alpha"), file = fout)
    for t in range(T):
        print("%-12d %-12d %-10.4f" %(SltWeakCls[t], SltThIdx[t], alpha[t]))
        print("%-12d %-12d %-10.4f" %(SltWeakCls[t], SltThIdx[t], alpha[t]), file = fout)
    
    # Check Training Positives
    # Final strong classifier: Use alpha, SltWeakCls, SltThIdx to construct
    #    Eq: np.sum(alpha * h(x)) - 0.5*sum(alpha) >= 0
    # Compute final pred, (2468,) shape 0-1 vector
    # Compute false-postive rate->FP/(FP+TN) should be 30% 
    # Compute true-postive rate->TP/(TP+FN) should be 99% or 1
    print("Testing on training set")
    print("Testing on training set", file = fout)
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
    print("TP, FP, TN, FN, TPR, FPR = (%d, %d, %d, %d, %.4f, %.4f)" %(TP, FP, TN, FN, TPR, FPR), file = fout)
    if FP == 0:  # If all images in the next stage belongs to correct category, there is no need to continue
        ContinueFlag = False
    else:
        ContinueFlag = True
    
    # Generating new masks
    Mask_ThisStage = list(Pred_Final==1)    # len = np.sum(Mask==1)
    Mask_TillNow = np.copy(Mask)            # len = 2468
    Mask_Postion = [idx for idx, x in enumerate(Mask) if x==True] # len = np.sum(Mask==1)
    for idx, current_mask in zip(Mask_Postion, Mask_ThisStage):
        Mask_TillNow[idx] = current_mask
    Mask_TillNow = np.array(Mask_TillNow==1)
    print("np.sum(Mask_TillNow) = ", np.sum(Mask_TillNow))
    np.save("Mask_epoch_%d.npy" %(Base_Stage+Stage), Mask_TillNow)
    
    # Also check in test test
    Feature_Final = np.zeros(TestGT.shape)
    Dec_Th = np.sum(alpha)/2
    print("Testing on test set")
    print("Testing on test set", file = fout)
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
    print("TP, FP, TN, FN, TPR, FPR = (%d, %d, %d, %d, %.4f, %.4f)" %(TP, FP, TN, FN, TPR, FPR), file = fout)
    print()
    fout.close()
    
    if ContinueFlag==False:
        break

