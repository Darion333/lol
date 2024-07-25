# High School Machine Learning Project
## Tutorial 1 - Using Classification Type Machine Learning to Differentiate Cancerous and Not Cancerous From Eachother
### Learning Objectives

  * Understand various commonly used machine learning models and algorithms and how they are utilized
  * Using a known set of cancerous and not cancerous molecules to train a given unknown set
    
### Background

Identifying whether or not molecules are cancerous is a very important task since many of them are used in various scientific studies and research. PAH molecules, or Polycyclic aromatic hydrocarbons are molecules that are made up of multiple carbon rings. While many PAH molecules are cancerous, others are not, highlighting the importance of knowing which molecules are which when used in research or experiments. 

Using a fundamental concept of chemistry, that similarly structured molecules will have similar properties, we can create a machine learning model that will compare the strucutures of given PAH molecules with other already classified molecules in order to determine which are cancerous.

### Code

```python
import os    
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import operator
import numpy as np
import sklearn.preprocessing
import sklearn.utils
from sklearn.decomposition import PCA 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, accuracy_score
import sklearn.metrics as sklm
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from functools import partial
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pickle
from matplotlib import rc
import matplotlib
import pandas as pd
from hyperopt import hp, tpe, fmin, Trials
from rdkit.Chem import AllChem
import pubchempy as pcp
from rdkit import Chem
```


