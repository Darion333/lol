# High School Machine Learning Project
## Tutorial 1 - Using Classification Type Machine Learning to Differentiate Cancerous and Not Cancerous From Eachother
### Learning Objectives

  * Use Scikit-learn to do simple classification tasks
    
### Background


Identifying cancerous molecules is crucial for scientific research, especially when these molecules are used in various studies. Polycyclic aromatic hydrocarbons (PAHs) are molecules formed by multiple carbon rings. While some PAH molecules are carcinogenic, others are not, underscoring the necessity of distinguishing between them in research settings.

Referencing a fundamental principle in chemistry, where molecules of similar structures exhibit similar properties, we can develop a machine learning model to classify PAH molecules. This model will compare the structures of unknown PAHs with those of molecules that have already been classified, predicting their carcinogenic potential.

We will use external libraries such as PubChemPy and RDKit. These tools will first represent the molecular structures as SMILES (Simplified Molecular Input Line Entry System) strings. The SMILES strings will then be converted into Morgan fingerprints, an array format that represents a molecule's structure in binary. This conversion is critical, as the molecular names stored as strings are insufficient for computational processing.

## Code:
&nbsp;  
### Set up
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

```python
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
```
* These code blocks are setting up the basics of what is needed for the model, importing necessary programming libraries and defining parameters for data tables

&nbsp;  
### Pre-processing the data set
&nbsp;  
```python
datapath = 'PAH/' # path to your data folder
filein_test= os.path.join(datapath,'testset_0.ds') # read in the CSV file containing the features. This file is just for example
filein_train= os.path.join(datapath,'trainset_0.ds')
# The dataframe for molecule name and classis
df_test = pd.read_csv(filein_test, sep=" ",  header=None, names=['molecule', 'cancerous'])
df_train = pd.read_csv(filein_train, sep=" ",  header=None, names=['molecule', 'cancerous'])
```
* This code block processes the data (different cancerous and uncancerous molecules) listed in a specified data folder
  &nbsp;  
```python
df_test
```
* This code block runs the function "df_test", outputting the data from a folder into a table with parameters defined by the previous code blocks, as seen below:
  
![image](https://github.com/user-attachments/assets/29aa6c34-9130-4316-925d-41c21857218b)

* This data is from the test set, in which the code will use as a reference while training the other unknown data

  &nbsp;  
```python
df_train
```
* This code block runs the function "df_train", outputting the data from a folder into a table with parameters defined by the previous code blocks, as seen below:

![image](https://github.com/user-attachments/assets/93b00e86-d1e6-421a-a88f-1a746b70c2b1)

* This data is from the train set, in which the code will change and train using the test data as a reference


&nbsp;  
```python
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
```
* This code block uses the external library PubChemPy, which is a way to utilize the world's largest collection of freely accessible chemical information, PubChem
* The name of each molecule will be found in the PubChem database, and then a SMILES string will be formed based on its molecular structure.
* The data is turned into a dictionary using the "to_dict" function, resulting in an output of the molecule (in this case napthalene) as a list of its elements and their positions, a partial representation seen below:

  ![image](https://github.com/user-attachments/assets/72262dff-79f7-4cf7-9e8b-e34d205a03d0)


   &nbsp;  
```python
getSMILES(df_train)
```
* This function converts all the molecules in the training set into SMILES strings

![image](https://github.com/user-attachments/assets/aed90f93-e0a1-4612-bd4c-31a5b5a64207)

&nbsp;  
```python
df_train
```
* This function outputs the training set data table with a new column for the molecules denoted as SMILES strings

![image](https://github.com/user-attachments/assets/859fbd5a-4213-4a36-b408-f9e1f9cf1eae)

&nbsp;  
```python
getSMILES(df_test)
```
* This function converts all the molecules in the testing set into SMILES strings

  ![image](https://github.com/user-attachments/assets/7b1c33da-e344-49c4-87d3-cfe487565905)

&nbsp;  
```python
df_test
```
* This function outputs the testing set data table with a new column for the molecules denoted as SMILES strings

  ![image](https://github.com/user-attachments/assets/af516147-6d01-4799-834e-6c68bc4e25f7)

&nbsp;  
```python
fpgen = AllChem.GetMorganGenerator(radius=2)
mol = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
fp = fpgen.GetFingerprintAsNumPy(mol)

for i in fp:
    print(i)
```
* This code block utilizes AllChem's Morgan Fingerprint generator to take the structural data processed from the SMILES strings, and denote it as an array of integers
&nbsp;  

```python
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
```
* This code block converts the SMILES strings from the test set into an array denoting its Morgan Fingerprint, and also converts its cancerous or uncancerous labels into a binary format
* The output can be seen below:

![image](https://github.com/user-attachments/assets/bc2ac740-4d0f-493f-9670-3843c31b956d)


&nbsp;  
```python
X_train,y_train = getData(df_train)
```

&nbsp;  
### Train a SVM and get the ROC curve
&nbsp;  
```python
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
```
* This code block imports the SVM module from scikit-learn which is then used to train the SVM classifier using the training data X_train and y_train
&nbsp;  
```python
clf.predict(X_test)
```
* This function uses the SVM to predict the cancerous labels of the Molecules in X test

![image](https://github.com/user-attachments/assets/db7c4102-c3d6-4811-841f-ea25ab4ec0c9)

&nbsp;  
```python
y_test
```
* y_test is the actual set of cancerous labels that apply to the molecules in the test set

![image](https://github.com/user-attachments/assets/6cb39244-23af-4bee-9981-791c8ae58bc4)

* A comparison between the predicted labels and the actual labels shows that the SVM isnt quite fully accurate
&nbsp;  

```python
from sklearn.metrics import RocCurveDisplay
svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
```
* This code block will plot the ROC (Receiver operating characteristic) Curve of the previously trained classifier, which is a graphical representation of the classifiers accuracy and performance

![image](https://github.com/user-attachments/assets/293d25ac-d197-41a3-baac-b8525f6d518b)

* The positive rate and false positive rate are plotted along the two different axes, the area under the curve denoting its accuracy

&nbsp;  
### Train a random forest and get the ROC curve
&nbsp;  
```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
```
![image](https://github.com/user-attachments/assets/6c055a1f-71e2-419e-8ac9-922d0caa1d40)

&nbsp;  
```python
from sklearn.linear_model import LogisticRegression

clf_lg = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.predict(X_test)
```
*
![image](https://github.com/user-attachments/assets/931d63f2-cb75-4c26-89a1-743e01f01aef)


&nbsp;  
```python
y_test
```
*
![image](https://github.com/user-attachments/assets/3a89a637-4c25-4c9e-98fe-565e0b0760d1)


&nbsp;  
### Upsampling
&nbsp;  
```python
from sklearn.utils import resample,shuffle
df_0 = df_train[df_train['cancerous'] == -1]
df_1 = df_train[df_train['cancerous'] == 1]

len(df_0), len(df_1)
```
![image](https://github.com/user-attachments/assets/177b530f-6adf-467a-a635-9f07d6907c9e)

&nbsp;  
```python
df_0_upsampled = resample(df_0,random_state=42,n_samples=50,replace=True)

len(df_0_upsampled)
```
![image](https://github.com/user-attachments/assets/958a4f8c-61f4-4e2f-bdee-48bc35bf04d6)

&nbsp;  
```python
df_0_upsampled
```
![image](https://github.com/user-attachments/assets/1ff23392-638a-47a1-a0f3-b22b6d5dcf81)
![image](https://github.com/user-attachments/assets/2cd41a27-0901-4c77-be3e-98b33127db9d)
![image](https://github.com/user-attachments/assets/65bb7e04-330c-4d88-bdb9-e2260ab3fab1)
&nbsp;  
```python
df_upsampled = pd.concat([df_0_upsampled,df_1])

X_train_up,y_train_up = getData(df_upsampled)

clf_lg = LogisticRegression(random_state=0).fit(X_train_up, y_train_up)
clf.predict(X_test)
```
![image](https://github.com/user-attachments/assets/29a39f13-e559-4e19-89d3-352e290deffe)
&nbsp;  


