# Watermelon feature selection

Repository of the feature selection method **watermelon**, original paper (on ICPR2020) available later.

## About watermelon
Watermelon is a feature selection method which scores the features through estimating the Bayes error rate based on kernel density estimation. Additionally, it updates the scores of features dynamically by quantitatively interpreting the effects of feature relevance and redundancy. Distinguishing from the common heuristic applied by many feature selection methods, which prefers choosing features that are not relevant to each other, watermelon penalizes only monotonically correlated features and rewards any other kind of relevance among features based on Spearman’s rank correlation coefficient and normalized mutual information.

## Installation
### Requirements
*Python 3.7*,
*pandas*,
*numpy*,
*scipy*,
*scikit-learn*

### Installation
Watermelon can be installed via pip from [PyPI](https://pypi.org/project/watermelon-feature-selection/)

```pip install watermelon-feature-selection```

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
    feature_indices,feature_score=watermelon_fs.watermelon(data,labels,n_select,par_cor,par_nmi)
```

Call the method in `if __name__ == "__main__"` block to use multiprocessing, for more information, see [here](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming)


You can find the project on [Github](https://github.com/Tzutori/watermelon-feature-selection)

## Contact
xiang.xie.china@gmail.com

more information will be updated soon.