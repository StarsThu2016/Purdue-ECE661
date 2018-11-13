# This is the utility functions for ECE 661 hw3
import numpy as np
import copy
def Homography(A,B):
    # Find the homography from A to B, for example
    #A = np.transpose(np.array([[1141,817,1], [1257,761,1], [1126,994,1], [1243,944,1]]))
    #B = np.transpose(np.array([[0,0,1], [60,0,1], [0,80,1], [60,80,1]]))
    Projection = np.zeros((8,1))   # 8 x 1
    Coefficient = np.zeros((8, 8))  # 8 x 8
    for index in range(4):
        Projection[index*2, 0] = B[0,index]    # xB
        Projection[index*2+1, 0] = B[1,index]  # yB
        Coefficient[index*2, 0] = A[0, index]  # xA
        Coefficient[index*2, 1] = A[1, index]  # yA
        Coefficient[index*2, 2] = 1
        Coefficient[index*2, 6] = - A[0, index] * B[0, index]  # -xA*xB
        Coefficient[index*2, 7] = - A[1, index] * B[0, index]  # -yA*xB
        Coefficient[index*2+1, 3] = A[0, index]  # xA
        Coefficient[index*2+1, 4] = A[1, index]  # yA
        Coefficient[index*2+1, 5] = 1
        Coefficient[index*2+1, 6] = - A[0, index] * B[1, index]  # -xA*yB
        Coefficient[index*2+1, 7] = - A[1, index] * B[1, index]  # -yA*yB
    h = np.matmul(np.linalg.inv(Coefficient), Projection) 
    H_AB = np.array(([h[0][0], h[1][0], h[2][0]], 
                     [h[3][0], h[4][0], h[5][0]], 
                     [h[6][0], h[7][0], 1]))
    return H_AB
    
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
