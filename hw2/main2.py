# This is the main python script for ECE 661 hw2
# To run: python3 main2.py
import numpy as np
from PIL import Image
from homography_util import *

A = np.array([[192,62], [406,110], [205,450], [413,383]])  
B = np.array([[54,54], [470,56], [59,466], [472,470]])
C = np.array([[126,107], [447,58], [119,418], [442,488]])
D = np.array([[0,0], [879,0], [0,719], [879,719]])
imgA = np.array(Image.open("PicsHw2/2-1.jpg"))  # 500W x 500H image 
imgB = np.array(Image.open("PicsHw2/2-2.jpg"))  # 500W x 500H image
imgC = np.array(Image.open("PicsHw2/2-3.jpg"))  # 500W x 500H image
imgD = np.array(Image.open("PicsHw2/2-4.jpg"))  # 880W x 720H image

# Task 1a
# Hymnographies and resulting images from Fig. 1(d) to Fig. 1(a)
H_DA = Homography(D,A)
print("H_DA = ", H_DA)
RefilledA_WithD = Filling(imgA, A, imgD, D)
RefilledImgA_WithD = Image.fromarray(RefilledA_WithD, 'RGB')
RefilledImgA_WithD.save("Task2a1.jpg")

# Homographies and resulting images from Fig. 1(d) to Fig. 1(b)
H_DB = Homography(D,B)
print("H_DB = ", H_DB)
RefilledB_WithD = Filling(imgB, B, imgD, D)
RefilledImgB_WithD = Image.fromarray(RefilledB_WithD, 'RGB')
RefilledImgB_WithD.save("Task2a2.jpg")

# Homographies and resulting images from Fig. 1(d) to Fig. 1(c)
H_DC = Homography(D,C)
print("H_DC = ", H_DC)
RefilledC_WithD = Filling(imgC, C, imgD, D)
RefilledImgC_WithD = Image.fromarray(RefilledC_WithD, 'RGB')
RefilledImgC_WithD.save("Task2a3.jpg")
print("----------")

# Task 1b
# Homographies from Fig. 1(a) to Fig. 1(b)
H_AB = Homography(A,B)
print("H_AB = ", H_AB)

# Homographies from Fig. 1(b) to Fig. 1(c)
H_BC = Homography(B,C)
print("H_BC = ", H_BC)

# Homographies from Fig. 1(a) to Fig. 1(c)
H_AC = Homography(A,C)
print("H_AC = ", H_AC)

H_ABC = np.matmul(H_BC, H_AB)
H_ABC = H_ABC/H_ABC[2][2]
print("H_ABC = ", H_ABC)

# Mapping the imgA to imgC plane
(x_min, y_min, mappedA) = Mapped(imgA, H_AC)
print("(x_min, y_min) = (%d, %d)" %(x_min, y_min))
mappedimgA = Image.fromarray(np.uint8(mappedA), 'RGB')
mappedimgA.save("Task2b.jpg")
