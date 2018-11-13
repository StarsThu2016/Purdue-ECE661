
# coding: utf-8

# In[55]:


import numpy as np
import cv2 as cv
def SSD(f1, f2):
    return np.sum((f1-f2)**2)
def NCC(f1, f2):
    m1 = np.mean(f1)
    m2 = np.mean(f2)
    top = np.sum((f1-m1)*(f2-m2))
    bottom = np.sqrt(np.sum((f1-m1)**2)*np.sum((f2-m2)**2))
    return top/bottom
def Correspondence(FM1, Coord1, FM2, Coord2, metric = "SSD"): 
    #(#,128), [i].pt=(2,), (##,128), [i].pt=(2,)
    N1 = FM1.shape[0]
    N2 = FM2.shape[0]
    Pairs = []
    Scores = []
    if metric=="SSD":
        for idx1 in range(N1):
            Score = np.array([SSD(FM1[idx1,:], FM2[idx2,:]) for idx2 in range(N2)])
            idx2 = np.argmin(Score)
            Scores.append(np.min(Score))
            Pairs.append((idx1,idx2))
        # Select 100 pairs
        BestPairs = [x for _,x in sorted(zip(Scores,Pairs))]  # Incresing order
    if metric=="NCC":
        for idx1 in range(N1):
            Score = np.array([NCC(FM1[idx1,:], FM2[idx2,:]) for idx2 in range(N2)])
            idx2 = np.argmax(Score)
            Scores.append(np.max(Score))
            Pairs.append((idx1,idx2)) 
        # Select 100 pairs
        BestPairs = [x for _,x in sorted(zip(Scores,Pairs), reverse = True)]  # Decresing order   
    return BestPairs[0:int(np.min([100,len(BestPairs)]))]   
def DrawPairs(input_image1, input_image2, Coord1, Coord2, Pairs, metric = "SSD"):
    left = cv.imread("HW4Pics/%s.jpg" %input_image1)
    right = cv.imread("HW4Pics/%s.jpg" %input_image2)
    output = np.concatenate((left, right), axis = 1)
    for (idx1, idx2) in Pairs:
        x1 = int(Coord1[idx1].pt[0])
        y1 = int(Coord1[idx1].pt[1])
        x2 = int(Coord2[idx2].pt[0])
        y2 = int(Coord2[idx2].pt[1]) + left.shape[1]
        cv.line(output, (y1,x1),(y2,x2), (0,0,255), 2)
    cv.imwrite("Coores_%s_%s.png" %(input_image1, metric), output) 


# In[61]:


# Input    
input_image1, input_image2 = ("1", "2")
img = cv.imread('HW4Pics/%s.jpg' %input_image1)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp1,img)
cv.imwrite('save_%s.jpg' %input_image1, img)

# Input    
img = cv.imread('HW4Pics/%s.jpg' %input_image2)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp2, des2 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp2,img)
cv.imwrite('save_%s.jpg' %input_image2, img)

Pairs = Correspondence(des1, kp1, des2, kp2, metric = "SSD")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "SSD")
Pairs = Correspondence(des1, kp1, des2, kp2, metric = "NCC")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "NCC")


# In[63]:


# Input    
input_image1, input_image2 = ("Truck1", "Truck2")
img = cv.imread('HW4Pics/%s.jpg' %input_image1)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp1,img)
cv.imwrite('save_%s.jpg' %input_image1, img)
print("Find %d key points" %len(kp1))

# Input    
img = cv.imread('HW4Pics/%s.jpg' %input_image2)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp2, des2 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp2,img)
cv.imwrite('save_%s.jpg' %input_image2, img)
print("Find %d key points" %len(kp2))

Pairs = Correspondence(des1, kp1, des2, kp2, metric = "SSD")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "SSD")
Pairs = Correspondence(des1, kp1, des2, kp2, metric = "NCC")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "NCC")


# In[64]:


# Input    
input_image1, input_image2 = ("Fountain1", "Fountain2")
img = cv.imread('HW4Pics/%s.jpg' %input_image1)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp1,img)
cv.imwrite('save_%s.jpg' %input_image1, img)
print("Find %d key points" %len(kp1))

# Input    
img = cv.imread('HW4Pics/%s.jpg' %input_image2)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp2, des2 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp2,img)
cv.imwrite('save_%s.jpg' %input_image2, img)
print("Find %d key points" %len(kp2))

Pairs = Correspondence(des1, kp1, des2, kp2, metric = "SSD")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "SSD")
Pairs = Correspondence(des1, kp1, des2, kp2, metric = "NCC")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "NCC")


# In[65]:


# Input    
input_image1, input_image2 = ("Tower1", "Tower2")
img = cv.imread('HW4Pics/%s.jpg' %input_image1)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp1,img)
cv.imwrite('save_%s.jpg' %input_image1, img)
print("Find %d key points" %len(kp1))

# Input    
img = cv.imread('HW4Pics/%s.jpg' %input_image2)
# SIFT Feature
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp2, des2 = sift.detectAndCompute(gray,None)
cv.drawKeypoints(img,kp2,img)
cv.imwrite('save_%s.jpg' %input_image2, img)
print("Find %d key points" %len(kp2))

Pairs = Correspondence(des1, kp1, des2, kp2, metric = "SSD")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "SSD")
Pairs = Correspondence(des1, kp1, des2, kp2, metric = "NCC")
DrawPairs(input_image1, input_image2, kp1, kp2, Pairs, metric = "NCC")

