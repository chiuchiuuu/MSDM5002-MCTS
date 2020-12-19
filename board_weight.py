import numpy as np

gauss_mat = np.zeros((15,15))
for i in range(15):
    for j in range(15):
        gauss_mat[i,j]=np.exp(-(((i-7)**2)+((j-7)**2)))/(2*np.pi)

gauss_mat = np.log10(gauss_mat)+50