import scipy.io
from sklearn.preprocessing import OneHotEncoder
from watermelon import watermelon
if __name__ == "__main__" :           
    par_cor=0.5
    par_nmi=0.5
    n_select=200
    
    mat = scipy.io.loadmat(r'..\data\colon.mat')
    data=mat['X']
    labels=mat['Y'].flatten()
    enc = OneHotEncoder(handle_unknown='ignore')
    labels=enc.fit_transform(labels).toarray()  
    
    watermelon_fs=watermelon.watermelon()
    feature_indices,feature_score=watermelon_fs.watermelon(data,labels,n_select,par_cor,par_nmi)