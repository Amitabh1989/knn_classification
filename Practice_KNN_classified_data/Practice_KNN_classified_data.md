# K Nearest Neighbors with Python
You've been given a classified data set from a company! They've hidden the feature column names but have given you the data and the target classes.

We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.

Let's grab it and use it!

# Import Libraries


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

# Import data


```python
df = pd.read_csv("Classified Data")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.913917</td>
      <td>1.162073</td>
      <td>0.567946</td>
      <td>0.755464</td>
      <td>0.780862</td>
      <td>0.352608</td>
      <td>0.759697</td>
      <td>0.643798</td>
      <td>0.879422</td>
      <td>1.231409</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.635632</td>
      <td>1.003722</td>
      <td>0.535342</td>
      <td>0.825645</td>
      <td>0.924109</td>
      <td>0.648450</td>
      <td>0.675334</td>
      <td>1.013546</td>
      <td>0.621552</td>
      <td>1.492702</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.721360</td>
      <td>1.201493</td>
      <td>0.921990</td>
      <td>0.855595</td>
      <td>1.526629</td>
      <td>0.720781</td>
      <td>1.626351</td>
      <td>1.154483</td>
      <td>0.957877</td>
      <td>1.285597</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.234204</td>
      <td>1.386726</td>
      <td>0.653046</td>
      <td>0.825624</td>
      <td>1.142504</td>
      <td>0.875128</td>
      <td>1.409708</td>
      <td>1.380003</td>
      <td>1.522692</td>
      <td>1.153093</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.279491</td>
      <td>0.949750</td>
      <td>0.627280</td>
      <td>0.668976</td>
      <td>1.232537</td>
      <td>0.703727</td>
      <td>1.115596</td>
      <td>0.646691</td>
      <td>1.463812</td>
      <td>1.419167</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# EDA


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Unnamed: 0    1000 non-null   int64  
     1   WTT           1000 non-null   float64
     2   PTI           1000 non-null   float64
     3   EQW           1000 non-null   float64
     4   SBI           1000 non-null   float64
     5   LQE           1000 non-null   float64
     6   QWG           1000 non-null   float64
     7   FDJ           1000 non-null   float64
     8   PJF           1000 non-null   float64
     9   HQE           1000 non-null   float64
     10  NXJ           1000 non-null   float64
     11  TARGET CLASS  1000 non-null   int64  
    dtypes: float64(10), int64(2)
    memory usage: 93.9 KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>499.500000</td>
      <td>0.949682</td>
      <td>1.114303</td>
      <td>0.834127</td>
      <td>0.682099</td>
      <td>1.032336</td>
      <td>0.943534</td>
      <td>0.963422</td>
      <td>1.071960</td>
      <td>1.158251</td>
      <td>1.362725</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288.819436</td>
      <td>0.289635</td>
      <td>0.257085</td>
      <td>0.291554</td>
      <td>0.229645</td>
      <td>0.243413</td>
      <td>0.256121</td>
      <td>0.255118</td>
      <td>0.288982</td>
      <td>0.293738</td>
      <td>0.204225</td>
      <td>0.50025</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.174412</td>
      <td>0.441398</td>
      <td>0.170924</td>
      <td>0.045027</td>
      <td>0.315307</td>
      <td>0.262389</td>
      <td>0.295228</td>
      <td>0.299476</td>
      <td>0.365157</td>
      <td>0.639693</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>249.750000</td>
      <td>0.742358</td>
      <td>0.942071</td>
      <td>0.615451</td>
      <td>0.515010</td>
      <td>0.870855</td>
      <td>0.761064</td>
      <td>0.784407</td>
      <td>0.866306</td>
      <td>0.934340</td>
      <td>1.222623</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>499.500000</td>
      <td>0.940475</td>
      <td>1.118486</td>
      <td>0.813264</td>
      <td>0.676835</td>
      <td>1.035824</td>
      <td>0.941502</td>
      <td>0.945333</td>
      <td>1.065500</td>
      <td>1.165556</td>
      <td>1.375368</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>749.250000</td>
      <td>1.163295</td>
      <td>1.307904</td>
      <td>1.028340</td>
      <td>0.834317</td>
      <td>1.198270</td>
      <td>1.123060</td>
      <td>1.134852</td>
      <td>1.283156</td>
      <td>1.383173</td>
      <td>1.504832</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999.000000</td>
      <td>1.721779</td>
      <td>1.833757</td>
      <td>1.722725</td>
      <td>1.634884</td>
      <td>1.650050</td>
      <td>1.666902</td>
      <td>1.713342</td>
      <td>1.785420</td>
      <td>1.885690</td>
      <td>1.893950</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```




    <AxesSubplot:>




    
![png](output_8_1.png)
    



```python
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull())
```




    <AxesSubplot:>




    
![png](output_9_1.png)
    


# Impressions

1. Data is pretty flat, meaning all are clean and no null values found
2. Unnamed row is not needed
3. Target column is that of a Classification stating 0 or 1
4. We can get directly into the KNN algorithm part.
5. Will do a pairplot as well for viewing more relation-ships.




```python
df.drop("Unnamed: 0", axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.913917</td>
      <td>1.162073</td>
      <td>0.567946</td>
      <td>0.755464</td>
      <td>0.780862</td>
      <td>0.352608</td>
      <td>0.759697</td>
      <td>0.643798</td>
      <td>0.879422</td>
      <td>1.231409</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.635632</td>
      <td>1.003722</td>
      <td>0.535342</td>
      <td>0.825645</td>
      <td>0.924109</td>
      <td>0.648450</td>
      <td>0.675334</td>
      <td>1.013546</td>
      <td>0.621552</td>
      <td>1.492702</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.721360</td>
      <td>1.201493</td>
      <td>0.921990</td>
      <td>0.855595</td>
      <td>1.526629</td>
      <td>0.720781</td>
      <td>1.626351</td>
      <td>1.154483</td>
      <td>0.957877</td>
      <td>1.285597</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.234204</td>
      <td>1.386726</td>
      <td>0.653046</td>
      <td>0.825624</td>
      <td>1.142504</td>
      <td>0.875128</td>
      <td>1.409708</td>
      <td>1.380003</td>
      <td>1.522692</td>
      <td>1.153093</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.279491</td>
      <td>0.949750</td>
      <td>0.627280</td>
      <td>0.668976</td>
      <td>1.232537</td>
      <td>0.703727</td>
      <td>1.115596</td>
      <td>0.646691</td>
      <td>1.463812</td>
      <td>1.419167</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1.010953</td>
      <td>1.034006</td>
      <td>0.853116</td>
      <td>0.622460</td>
      <td>1.036610</td>
      <td>0.586240</td>
      <td>0.746811</td>
      <td>0.319752</td>
      <td>1.117340</td>
      <td>1.348517</td>
      <td>1</td>
    </tr>
    <tr>
      <th>996</th>
      <td>0.575529</td>
      <td>0.955786</td>
      <td>0.941835</td>
      <td>0.792882</td>
      <td>1.414277</td>
      <td>1.269540</td>
      <td>1.055928</td>
      <td>0.713193</td>
      <td>0.958684</td>
      <td>1.663489</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1.135470</td>
      <td>0.982462</td>
      <td>0.781905</td>
      <td>0.916738</td>
      <td>0.901031</td>
      <td>0.884738</td>
      <td>0.386802</td>
      <td>0.389584</td>
      <td>0.919191</td>
      <td>1.385504</td>
      <td>1</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1.084894</td>
      <td>0.861769</td>
      <td>0.407158</td>
      <td>0.665696</td>
      <td>1.608612</td>
      <td>0.943859</td>
      <td>0.855806</td>
      <td>1.061338</td>
      <td>1.277456</td>
      <td>1.188063</td>
      <td>1</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0.837460</td>
      <td>0.961184</td>
      <td>0.417006</td>
      <td>0.799784</td>
      <td>0.934399</td>
      <td>0.424762</td>
      <td>0.778234</td>
      <td>0.907962</td>
      <td>1.257190</td>
      <td>1.364837</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 11 columns</p>
</div>




```python
df.corrwith(df["TARGET CLASS"]).sort_values().plot(kind='bar')
```




    <AxesSubplot:>




    
![png](output_12_1.png)
    


# Build the model


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
```


```python
from sklearn.model_selection import train_test_split
```


```python
from sklearn.preprocessing import StandardScaler
```


```python
X = df.drop(['TARGET CLASS', 'Unnamed: 0'], axis=1)
```


```python
y = df['TARGET CLASS']
```


```python
scaler = StandardScaler()
```


```python
scaler.fit(X)
```




    StandardScaler()




```python
scaled_data = scaler.transform(X)
```


```python
scaled_data_df = pd.DataFrame(scaled_data, columns=X.columns)
```


```python
 X.columns
```




    Index(['WTT', 'PTI', 'EQW', 'SBI', 'LQE', 'QWG', 'FDJ', 'PJF', 'HQE', 'NXJ'], dtype='object')




```python
scaled_data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.123542</td>
      <td>0.185907</td>
      <td>-0.913431</td>
      <td>0.319629</td>
      <td>-1.033637</td>
      <td>-2.308375</td>
      <td>-0.798951</td>
      <td>-1.482368</td>
      <td>-0.949719</td>
      <td>-0.643314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.084836</td>
      <td>-0.430348</td>
      <td>-1.025313</td>
      <td>0.625388</td>
      <td>-0.444847</td>
      <td>-1.152706</td>
      <td>-1.129797</td>
      <td>-0.202240</td>
      <td>-1.828051</td>
      <td>0.636759</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.788702</td>
      <td>0.339318</td>
      <td>0.301511</td>
      <td>0.755873</td>
      <td>2.031693</td>
      <td>-0.870156</td>
      <td>2.599818</td>
      <td>0.285707</td>
      <td>-0.682494</td>
      <td>-0.377850</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.982841</td>
      <td>1.060193</td>
      <td>-0.621399</td>
      <td>0.625299</td>
      <td>0.452820</td>
      <td>-0.267220</td>
      <td>1.750208</td>
      <td>1.066491</td>
      <td>1.241325</td>
      <td>-1.026987</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.139275</td>
      <td>-0.640392</td>
      <td>-0.709819</td>
      <td>-0.057175</td>
      <td>0.822886</td>
      <td>-0.936773</td>
      <td>0.596782</td>
      <td>-1.472352</td>
      <td>1.040772</td>
      <td>0.276510</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>0.211653</td>
      <td>-0.312490</td>
      <td>0.065163</td>
      <td>-0.259834</td>
      <td>0.017567</td>
      <td>-1.395721</td>
      <td>-0.849486</td>
      <td>-2.604264</td>
      <td>-0.139347</td>
      <td>-0.069602</td>
    </tr>
    <tr>
      <th>996</th>
      <td>-1.292453</td>
      <td>-0.616901</td>
      <td>0.369613</td>
      <td>0.482648</td>
      <td>1.569891</td>
      <td>1.273495</td>
      <td>0.362784</td>
      <td>-1.242110</td>
      <td>-0.679746</td>
      <td>1.473448</td>
    </tr>
    <tr>
      <th>997</th>
      <td>0.641777</td>
      <td>-0.513083</td>
      <td>-0.179205</td>
      <td>1.022255</td>
      <td>-0.539703</td>
      <td>-0.229680</td>
      <td>-2.261339</td>
      <td>-2.362494</td>
      <td>-0.814261</td>
      <td>0.111597</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0.467072</td>
      <td>-0.982786</td>
      <td>-1.465194</td>
      <td>-0.071465</td>
      <td>2.368666</td>
      <td>0.001269</td>
      <td>-0.422041</td>
      <td>-0.036777</td>
      <td>0.406025</td>
      <td>-0.855670</td>
    </tr>
    <tr>
      <th>999</th>
      <td>-0.387654</td>
      <td>-0.595894</td>
      <td>-1.431398</td>
      <td>0.512722</td>
      <td>-0.402552</td>
      <td>-2.026512</td>
      <td>-0.726253</td>
      <td>-0.567789</td>
      <td>0.336997</td>
      <td>0.010350</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 10 columns</p>
</div>




```python
knn = KNeighborsClassifier(n_neighbors=1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(n_neighbors=1)




```python
y_pred = knn.predict(X_test)
```


```python
plt.plot(y_test-y_pred)
```




    [<matplotlib.lines.Line2D at 0x1f4c4acb9d0>]




    
![png](output_29_1.png)
    



```python
plt.plot(y_test, y_pred)
```




    [<matplotlib.lines.Line2D at 0x1f4c7070640>]




    
![png](output_30_1.png)
    



```python
# sns.scatterplot(x=y_test, hue=y_pred, data=df)
```


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.92      0.94      0.93       159
               1       0.93      0.91      0.92       141
    
        accuracy                           0.93       300
       macro avg       0.93      0.93      0.93       300
    weighted avg       0.93      0.93      0.93       300
    
    


```python
from sklearn.metrics import confusion_matrix
```


```python
confusion_matrix(y_test, y_pred)
```




    array([[150,   9],
           [ 13, 128]], dtype=int64)



# Explore other K values for better results

Now let's see if we can get a better score than this. The current one at K=1 is pretty good. But can we get better value if we choose any other value for K


```python
# We will compute error rate here. Meaning, we will take the MEAN of all the values where the Predicted value was not equal
# to the y_test val. This will be done for all the K values from 1 to 40

error_rate = []

for k in range(1, 41):
    print(">>> Checking with K val as {}".format(k))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))
    
```

    >>> Checking with K val as 1
    >>> Checking with K val as 2
    >>> Checking with K val as 3
    >>> Checking with K val as 4
    >>> Checking with K val as 5
    >>> Checking with K val as 6
    >>> Checking with K val as 7
    >>> Checking with K val as 8
    >>> Checking with K val as 9
    >>> Checking with K val as 10
    >>> Checking with K val as 11
    >>> Checking with K val as 12
    >>> Checking with K val as 13
    >>> Checking with K val as 14
    >>> Checking with K val as 15
    >>> Checking with K val as 16
    >>> Checking with K val as 17
    >>> Checking with K val as 18
    >>> Checking with K val as 19
    >>> Checking with K val as 20
    >>> Checking with K val as 21
    >>> Checking with K val as 22
    >>> Checking with K val as 23
    >>> Checking with K val as 24
    >>> Checking with K val as 25
    >>> Checking with K val as 26
    >>> Checking with K val as 27
    >>> Checking with K val as 28
    >>> Checking with K val as 29
    >>> Checking with K val as 30
    >>> Checking with K val as 31
    >>> Checking with K val as 32
    >>> Checking with K val as 33
    >>> Checking with K val as 34
    >>> Checking with K val as 35
    >>> Checking with K val as 36
    >>> Checking with K val as 37
    >>> Checking with K val as 38
    >>> Checking with K val as 39
    >>> Checking with K val as 40
    


```python
plt.figure(figsize=(10, 8))
plt.plot(error_rate, linestyle="--", markeredgecolor='red', marker='o', markersize=10, markerfacecolor='red')
```




    [<matplotlib.lines.Line2D at 0x1f4c8399f40>]




    
![png](output_37_1.png)
    


# Impressions

We can see that at around K = 13 or 15, the error rate comes down to 0.05. We also see that once we keep going further up with the K value, at around k = 31, error rate gets pretty low. But I will not choose that as once we increase the rate, we see some quick jump off to higher error rate. But as it goes, you can always try with those values and figure out, what works the best for you.

So for this notebook, let's basically see what accuracies we get from K = 15


```python
knn = KNeighborsClassifier(n_neighbors=15)
```


```python
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(n_neighbors=15)




```python
y_pred = knn.predict(X_test)
```


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.94      0.95      0.95       159
               1       0.94      0.94      0.94       141
    
        accuracy                           0.94       300
       macro avg       0.94      0.94      0.94       300
    weighted avg       0.94      0.94      0.94       300
    
    


```python
# As you can see, we did marginally better with k = 15. This tempts me to try with K = 31. So here we go.
```


```python
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.96      0.96       159
               1       0.95      0.95      0.95       141
    
        accuracy                           0.95       300
       macro avg       0.95      0.95      0.95       300
    weighted avg       0.95      0.95      0.95       300
    
    

I feel alughing out loud. Actually I did while tying this. We see much better accuracy here with K = 31. But as stands
the truth, its all a matter of trail. Choose what works the best for you.


# Kudos and Enjoy !
