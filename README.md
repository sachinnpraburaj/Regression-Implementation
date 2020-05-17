# Regression-Implementation
This repo contains implementations of univariate and multivariate regression in Machine Learning with polynomial and modified relu basis functions along with and without L2 regularization.

**SOWC_combined_simple.csv**
- Dataset for the implementation    

**data_utils.py**
- Base code containing implementations for
    - loading and cleaning the data
    - z-normalizing (standardizing) features
    - modified relu basis function
    - computing design matrix for polynomial and relu basis functios using data
    - estimating weights for learning with and without regularization
    - evaluating the regressor
    
**polynomial_regression.py**
- Using base code to perform unregularized multivariate polynomial regressions of degree 1-6

**polynomial_regression_1d.py**
- Using base code to perform unregularized univariate polynomial regression of degree 3

**visualize_1d.py**
- To visualize the learned curve

**relu_regression.py**
- Using base code to perform unregularized multivariate relu regression

**polynomial_regression_reg.py**
- Using base code to perform L2 regularized multivariate polynomial regression of degree 2
- Identifying the best regularization constant using 10 fold cross-validation implementation

Folders **inv** and **pinv** contain visualizations.

**For execution:**
```
python3 filename.py
```



