# Watermelon feature selection

This is the python implementation of the feature selection method **watermelon**, original paper (on ICPR2020) available later.

## Code example

```python
import scipy.io
from watermelon import watermelon
if __name__ == "__main__" :           
    par_cor=0.5
    par_nmi=0.5
    n_select=200
    
    mat = scipy.io.loadmat(r'..\data\colon.mat')
    data=mat['X']
    labels=mat['Y'].flatten()
        
    watermelon_fs=watermelon.watermelon()
    feature_indice,feature_score=watermelon_fs.watermelon(data,labels,n_select,par_cor,par_nmi)
```


You can find the project on [Github](https://github.com/Tzutori/watermelon-feature-selection)

more information will be updated soon.