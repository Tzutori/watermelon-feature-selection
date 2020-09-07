import scipy.io
from watermelon import watermelon
if __name__ == "__main__" :           
    par_cor=0.5
    par_nmi=0.3
    n_select=200
    
    mat = scipy.io.loadmat(r'..\data\colon.mat')
    data=mat['X']
    labels=mat['Y'].flatten()
        
    fs_model=watermelon.watermelon()
    bie_result,bie_score=fs_model.watermelon(data,labels,n_select,par_cor,par_nmi,ovo=True,performance_metric='class balance',min_kde_bandwidth=0.9,verbose=True)