# This is the utility functions for ECE 661 hw2
import numpy as np
import copy
def Homography(A,B):
    # Find the homography from A to B, for example
    # A = np.array([[1467,75], [2985,681], [1440,2340], [3048,2094]])  
    # B = np.array([[1278,258], [3054,573], [1245,2082], [3081,1938]])
    Projection = np.reshape(B, (B.shape[0]*B.shape[1],1))  # (xB1,yB1,xB2,yB2,...)
    Coefficient = np.zeros((A.shape[0]*2, 8))  # number of points *2 x 8
    for index in range(A.shape[0]):
        Coefficient[index*2, 0] = A[index, 0]  # xA
        Coefficient[index*2, 1] = A[index, 1]  # yA
        Coefficient[index*2, 2] = 1
        Coefficient[index*2, 6] = - A[index, 0] * B[index, 0]  # -xA*xB
        Coefficient[index*2, 7] = - A[index, 1] * B[index, 0]  # -yA*xB
        Coefficient[index*2+1, 3] = A[index, 0] # xA
        Coefficient[index*2+1, 4] = A[index, 1] # yA
        Coefficient[index*2+1, 5] = 1
        Coefficient[index*2+1, 6] = - A[index, 0] * B[index, 1]  # -xA*yB
        Coefficient[index*2+1, 7] = - A[index, 1] * B[index, 1]  # -yA*yB
    h = np.matmul(np.linalg.inv(Coefficient), Projection) 
    H_DA = np.array(([h[0][0], h[1][0], h[2][0]], 
                     [h[3][0], h[4][0], h[5][0]], 
                     [h[6][0], h[7][0], 1]))
    return H_DA
    
def Filling(imgA, A, imgB, B):
    # Fill in the area of A in imgA with the area of B in imgB
    # Note: B must be a rectangular area!
    # A = np.array([[1467,75], [2985,681], [1440,2340], [3048,2094]]) 
    # B = np.array([[0,0], [1280,0], [0,720], [1280,720]])
    # imgA could be a (2709, 3612, 3) numpy.ndarray
    # imgB could be a (720, 1280, 3) numpy.ndarray  
    H_AB = Homography(A,B)
    
    # Construct a 3-D coordinate matrix of A
    # [ 0, 0, 0, ........., 1, 1, 1, ........., 2 ..]  
    # [ 0, 1, 2, ..., 2708, 0, 1, 2, ..., 2708, ... ]  
    # [ 1, 1, 1, 1, 1, 1, 1, ...................,1  ]
    # Meaning
    # [ X coordinate = Width direction, e.g. 3612]
    # [ Y coordinate = Height direction, e.g. 2709]
    # [ Ones ]
    HC_A = np.ones((3, imgA.shape[0]*imgA.shape[1])).astype(int)
    for indexH in range(imgA.shape[1]): 
        HC_A[0, indexH*imgA.shape[0]:indexH*imgA.shape[0]+imgA.shape[0]] = \
            np.repeat([indexH], imgA.shape[0])
        HC_A[1, indexH*imgA.shape[0]:indexH*imgA.shape[0]+imgA.shape[0]] = \
            np.arange(imgA.shape[0])
    HC_MappedA = np.matmul(H_AB, HC_A)  
    HC_MappedA = np.round(HC_MappedA/HC_MappedA[2,:]).astype(int) 
    
    # Check what mapped cooredinates is inside B
    EditVector = np.logical_and(HC_MappedA[0,:]>=B[0,0], HC_MappedA[0,:]<=B[1,0])
    EditVector = np.logical_and(HC_MappedA[1,:]>=B[0,1], EditVector)
    EditVector = np.logical_and(HC_MappedA[1,:]<=B[2,1], EditVector)
    
    # Refill the image
    # Map all pixels from imgA plane to imgB plane, replace the pixels that mapped to B
    RefilledA_WithB = copy.deepcopy(imgA)
    for index in np.arange(imgA.shape[0]*imgA.shape[1])[EditVector]:
        RefilledA_WithB[HC_A[1,index], HC_A[0,index], :] = \
            imgB[HC_MappedA[1,index], HC_MappedA[0,index], :]
    return RefilledA_WithB

def Mapped(imgA, H):   
    # Boundary mapping
    HC_Boundary_A = np.array([[0,imgA.shape[1]-1,0,imgA.shape[1]-1], 
                              [0,0,imgA.shape[0]-1,imgA.shape[0]-1], 
                              [1,1,1,1]]).astype(int)
    HC_Boundary_MappedA = np.matmul(H, HC_Boundary_A)  
    HC_Boundary_MappedA = (HC_Boundary_MappedA/HC_Boundary_MappedA[2,:]).astype(int)
    
    x_min = np.min(HC_Boundary_MappedA[0,:])
    x_max = np.max(HC_Boundary_MappedA[0,:])
    y_min = np.min(HC_Boundary_MappedA[1,:])
    y_max = np.max(HC_Boundary_MappedA[1,:])
    x_lim = x_max-x_min+1
    y_lim = y_max-y_min+1
     
    H_inverse = np.linalg.inv(H)
    # Construct a 3-D coordinate matrix "HC_mappedA" of mapped A (real HC)
    # [ 0, 0, 0, ........., 1, 1, 1, ........., (x_lim-1) ..] + x_min   
    # [ 0, 1, 2, ..., (y_lim-1), 0, 1, 2, ..., (y_lim-1), ... ] + y_min 
    # [ 1, 1, 1, 1, 1, 1, 1, ...................,1  ]
    # Meaning
    # [ X coordinate = Width direction, e.g. x_lim]
    # [ Y coordinate = Height direction, e.g. y_lim]
    # [ Ones ]
    HC_mappedA = np.ones((3, x_lim*y_lim)).astype(int)
    for indexH in range(x_lim): 
        HC_mappedA[0, indexH*y_lim:indexH*y_lim+y_lim] = \
            np.repeat([indexH], y_lim) + x_min
        HC_mappedA[1, indexH*y_lim:indexH*y_lim+y_lim] = \
            np.arange(y_lim) + y_min
    HC_A = np.matmul(H_inverse, HC_mappedA)  
    HC_A = (np.round(HC_A/HC_A[2,:])).astype(int) 
    #print(np.max(HC_A[1,:])) 
    
    A = np.array([[0,0], [imgA.shape[0],0], 
                  [0,imgA.shape[1]], [imgA.shape[0],imgA.shape[1]]]) 
    # Check what mapped cooredinates is inside A
    EditVector = np.logical_and(HC_A[0,:]>=HC_Boundary_A[0,0], HC_A[0,:]<=HC_Boundary_A[0,1])
    EditVector = np.logical_and(HC_A[1,:]>=HC_Boundary_A[1,0], EditVector)
    EditVector = np.logical_and(HC_A[1,:]<=HC_Boundary_A[1,2], EditVector)
    
    # Refill the image
    # Map all pixels inversely from mappedA plane to imgA plane, replace the pixels that mapped to A
    mappedA = np.zeros((y_lim,x_lim,3)).astype(int)
    for index in np.arange(x_lim*y_lim)[EditVector]:
        mappedA[HC_mappedA[1,index]-y_min, HC_mappedA[0,index]-x_min, :] = \
            imgA[HC_A[1,index], HC_A[0,index], :]    
    return (x_min,y_min,mappedA)
