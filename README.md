# High School Machine Learning Project
## Tutorial 1 - Using Classification Type Machine Learning to Differentiate Cancerous and Not Cancerous From Eachother
### Learning Objectives

  * Understand various commonly used machine learning classification methods and algorithms and how they are utilized
  * Using a known set of cancerous and not cancerous molecules to train a given unknown set
    
### Background

Identifying whether or not molecules are cancerous is a very important task since many of them are used in various scientific studies and research. PAH molecules, or Polycyclic aromatic hydrocarbons are molecules that are made up of multiple carbon rings. While many PAH molecules are cancerous, others are not, highlighting the importance of knowing which molecules are which when used in research or experiments. 

Using a fundamental concept of chemistry, that similarly structured molecules will have similar properties, we can create a machine learning model that will compare the strucutures of given PAH molecules with other already classified molecules in order to determine which are cancerous.

This code will use outside external models such as PubChemPy and Rdkit in order to first store the molecule name as a SMILES (Simplified molecular-input line-entry system) string which will then be converted into array (Morgan fingerprint), a process that is vital since we are unable to only use the name of the molecule stored as a string in the code.

### Code:

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

fs = 10 # font size
fs_label = 10 # tick label size
fs_lgd = 10 # legend font size
ss = 20 # symbol size
ts = 3 # tick size
slw = 1 # symbol line width
framelw = 1 # line width of frame
lw = 2 # line width of the bar box
rc('axes', linewidth=framelw)
plt.rcParams.update({
    "text.usetex": False,
    "font.weight":"bold",
    "axes.labelweight":"bold",
    "font.size":fs,
    'pdf.fonttype':'truetype'
})
plt.rcParams['mathtext.fontset']='stix'

datapath = 'PAH/' # path to your data folder
filein_test= os.path.join(datapath,'testset_0.ds') # read in the CSV file containing the features. This file is just for example
filein_train= os.path.join(datapath,'trainset_0.ds')
# The dataframe for molecule name and classis
df_test = pd.read_csv(filein_test, sep=" ",  header=None, names=['molecule', 'cancerous'])
df_train = pd.read_csv(filein_train, sep=" ",  header=None, names=['molecule', 'cancerous'])

df_test

df_train

def getSMILES(df):
    mols=df['molecule'].values
    smiles_list = []
    for mol in mols:
        # Get rid of the ".ct" suffix
        # Search Pubchem by the compound name
        results = pcp.get_compounds(mol[:-3], 'name')
        smiles = ""
        if len(results) > 0:
            # Get the SMILES string of the compound
            smiles = results[0].isomeric_smiles
            smiles_list.append(smiles)
            print(mol[:-3],smiles)
        else:
            smiles_list.append(smiles)
            print(mol[:-3],'molecule not found in PubChem')
    df['SMILES'] = smiles_list

mymol = pcp.get_compounds('naphthalene', 'name', record_type='3d')[0]

mydict=mymol.to_dict(properties=['atoms'])

mydict['atoms']

getSMILES(df_train)

df_train

getSMILES(df_test)

df_test

fpgen = AllChem.GetMorganGenerator(radius=2)
mol = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
fp = fpgen.GetFingerprintAsNumPy(mol)

for i in fp:
    print(i)

def getData(df):
    fpgen = AllChem.GetMorganGenerator(radius=2)
    MFP_list = []
    for smiles in df['SMILES'].values:
        mol = Chem.MolFromSmiles(smiles)
        MFP = fpgen.GetFingerprintAsNumPy(mol)
        MFP_list.append(MFP)
    X = np.array(MFP_list)

    y_list = []
    for y in df['cancerous']:
        if y == 1:
            y_list.append(1)
        else:
            y_list.append(0)
    y = np.array(y_list)
    return X,y

X_test,y_test = getData(df_test)

print(X_test)

X_train,y_train = getData(df_train)

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

clf.predict(X_test)

y_test

from sklearn.metrics import RocCurveDisplay
svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()

from sklearn.linear_model import LogisticRegression

clf_lg = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.predict(X_test)

y_test

from sklearn.utils import resample,shuffle
df_0 = df_train[df_train['cancerous'] == -1]
df_1 = df_train[df_train['cancerous'] == 1]

len(df_0), len(df_1)

df_0_upsampled = resample(df_0,random_state=42,n_samples=50,replace=True)

len(df_0_upsampled)

df_0_upsampled

df_upsampled = pd.concat([df_0_upsampled,df_1])

X_train_up,y_train_up = getData(df_upsampled)

clf_lg = LogisticRegression(random_state=0).fit(X_train_up, y_train_up)
clf.predict(X_test)

```
### Conclusion

Knowing whether or not molecules are cancerous is very important in many different scientific studies and research. Using machine learning and many of its different classification methods, we are able to compare the structures of a known set of molecules (training set) to a set of molecules that we dont know if cancerous or not yet in order to determine whether it is or not.

