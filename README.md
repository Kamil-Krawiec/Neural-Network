# Neural-Network
# Lab01

```python
X = heart_disease.data.features
y = heart_disease.data.targets
```

## Ogólny przegląd danych

Additional Information

This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.  In particular, the Cleveland database is the only one that has been used by ML researchers to date.  The "goal" field refers to the presence of heart disease in the patient.  It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

wartości liczbowe atrybutu num:
    -0 brak oznak choroby
    -1,2,3,4 występujące oznaki choroby


```python
# variable information
variable_info = heart_disease.variables
variable_info
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
      <th>name</th>
      <th>role</th>
      <th>type</th>
      <th>demographic</th>
      <th>description</th>
      <th>units</th>
      <th>missing_values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>Age</td>
      <td>None</td>
      <td>years</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sex</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>Sex</td>
      <td>None</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cp</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>trestbps</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>resting blood pressure (on admission to the ho...</td>
      <td>mm Hg</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>chol</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>serum cholestoral</td>
      <td>mg/dl</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>fbs</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>fasting blood sugar &gt; 120 mg/dl</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>restecg</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>thalach</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>maximum heart rate achieved</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>exang</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>exercise induced angina</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>oldpeak</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>ST depression induced by exercise relative to ...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>10</th>
      <td>slope</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ca</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>number of major vessels (0-3) colored by flour...</td>
      <td>None</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>thal</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>num</td>
      <td>Target</td>
      <td>Integer</td>
      <td>None</td>
      <td>diagnosis of heart disease</td>
      <td>None</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
X
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>3</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>3.0</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>45</td>
      <td>1</td>
      <td>1</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>1.2</td>
      <td>2</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>68</td>
      <td>1</td>
      <td>4</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>0</td>
      <td>141</td>
      <td>0</td>
      <td>3.4</td>
      <td>2</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>57</td>
      <td>1</td>
      <td>4</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>2</td>
      <td>1.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>2</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>38</td>
      <td>1</td>
      <td>3</td>
      <td>138</td>
      <td>175</td>
      <td>0</td>
      <td>0</td>
      <td>173</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 13 columns</p>
</div>




```python
y
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
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>2</td>
    </tr>
    <tr>
      <th>300</th>
      <td>3</td>
    </tr>
    <tr>
      <th>301</th>
      <td>1</td>
    </tr>
    <tr>
      <th>302</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 1 columns</p>
</div>



## 1.Czy zbiór jest zbalansowany pod względem liczby próbek na klasy?


```python
y_value_counts = y.value_counts()
y_value_counts
```




    num
    0      164
    1       55
    2       36
    3       35
    4       13
    Name: count, dtype: int64




```python
y_value_counts.plot(kind='barh')
```




    <Axes: ylabel='num'>




    
![png](../media/Lab01_files/Lab02_11_1.png)
    



```python
y_value_counts.plot(kind='pie', autopct='%1.1f%%')
```




    <Axes: ylabel='count'>




    
![png](../media/Lab01_files/Lab02_12_1.png)
    


Można zauważyć, że najwięcej przypadków jest dla wartości 0, która stanowi ponad połowę wszystkich wartości, jeśli chodzi o pozostałe, przypadek 1 posiada również duży wkład, 2 i 3 mają prawie taką samą częstotliwość na poziomie ok.12%, przypadek 4 jest najmniej liczny i stanowi niecałe 5%.

Odpowiedź:
Zbiór danych nie jest najlepiej zbalansowany, ponieważ niektóre klasy mają znacznie więcej próbek niż inne.


```python
no_presence = y[y==0].count().sum()
presence = y[y!=0].count().sum()
presence,no_presence
```




    (139, 164)




```python
# Create a bar chart
plt.bar(['No Presence', 'Presence'], [no_presence, presence])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Bar Chart: No Presence vs. Presence')
```




    Text(0.5, 1.0, 'Bar Chart: No Presence vs. Presence')




    
![png](../media/Lab01_files/Lab02_16_1.png)
    



```python
labels = ['No Presence', 'Presence']
sizes = [no_presence, presence]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Pie Chart: No Presence vs. Presence')
```




    Text(0.5, 1.0, 'Pie Chart: No Presence vs. Presence')




    
![png](../media/Lab01_files/Lab02_17_1.png)
    


Jeśli jednak pójść dalej i zobaczyć na wartości atrybutu num w perspektywie - 'ma objawy' 'nie ma objawow', rozkład będzie bardziej zbalansowany.

## 2. Jakie są średnie i odchylenia cech liczbowych?


```python
numeric_variables = variable_info[(variable_info['type']=='Integer') & (variable_info['name']!='num')]['name']
numeric_variables
```




    0          age
    3     trestbps
    4         chol
    7      thalach
    9      oldpeak
    11          ca
    Name: name, dtype: object




```python
X[numeric_variables].describe().loc[['mean','std']]
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
      <th>age</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>thalach</th>
      <th>oldpeak</th>
      <th>ca</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>54.438944</td>
      <td>131.689769</td>
      <td>246.693069</td>
      <td>149.607261</td>
      <td>1.039604</td>
      <td>0.672241</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.038662</td>
      <td>17.599748</td>
      <td>51.776918</td>
      <td>22.875003</td>
      <td>1.161075</td>
      <td>0.937438</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Dla cech liczbowych: czy ich rozkład jest w przybliżeniu normalny?


```python
from scipy import stats
```


```python
numeric_df = X[numeric_variables]
numeric_df
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
      <th>age</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>thalach</th>
      <th>oldpeak</th>
      <th>ca</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>145</td>
      <td>233</td>
      <td>150</td>
      <td>2.3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>160</td>
      <td>286</td>
      <td>108</td>
      <td>1.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>120</td>
      <td>229</td>
      <td>129</td>
      <td>2.6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>130</td>
      <td>250</td>
      <td>187</td>
      <td>3.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>130</td>
      <td>204</td>
      <td>172</td>
      <td>1.4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>45</td>
      <td>110</td>
      <td>264</td>
      <td>132</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>68</td>
      <td>144</td>
      <td>193</td>
      <td>141</td>
      <td>3.4</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>57</td>
      <td>130</td>
      <td>131</td>
      <td>115</td>
      <td>1.2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>130</td>
      <td>236</td>
      <td>174</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>38</td>
      <td>138</td>
      <td>175</td>
      <td>173</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 6 columns</p>
</div>




```python
# Perform the Shapiro-Wilk test and create histograms for each attribute
for column in numeric_df.columns:
    # Shapiro-Wilk test
    p_value = stats.shapiro(numeric_df[column])[1]

    # Create a histogram
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(numeric_df[column], bins=15, color='blue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # Check normality based on p-value
    plt.subplot(1, 2, 2)
    if p_value > 0.05:
        plt.text(0.1, 0.5, f'p-value: {p_value:.4f}\nProbably Normal', fontsize=12)
    else:
        plt.text(0.1, 0.5, f'p-value: {p_value:.4f}\nNot Normal', fontsize=12, color='red')
    plt.axis('off')
    plt.title('Shapiro-Wilk Test')

    plt.tight_layout()
    plt.show()
```


    
![png](../media/Lab01_files/Lab02_25_0.png)
    



    
![png](../media/Lab01_files/Lab02_25_1.png)
    



    
![png](../media/Lab01_files/Lab02_25_2.png)
    



    
![png](../media/Lab01_files/Lab02_25_3.png)
    



    
![png](../media/Lab01_files/Lab02_25_4.png)
    



    
![png](../media/Lab01_files/Lab02_25_5.png)
    



```python
import statsmodels.api as sm
```


```python
num_bins = 15

# Iterate through each column and create histograms and QQ plots
for column in numeric_df.columns:
    # Create a figure with subplots (histogram and QQ plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Create a histogram
    ax1.hist(numeric_df[column], bins=num_bins, color='blue', edgecolor='black')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram')

    # Create a QQ plot for the transformed data
    sm.qqplot(numeric_df[column], line='s', ax=ax2)
    ax2.set_title('QQ Plot')

    plt.tight_layout()
    plt.show()
```


    
![png](../media/Lab01_files/Lab02_27_0.png)
    



    
![png](../media/Lab01_files/Lab02_27_1.png)
    



    
![png](../media/Lab01_files/Lab02_27_2.png)
    



    
![png](../media/Lab01_files/Lab02_27_3.png)
    



    
![png](../media/Lab01_files/Lab02_27_4.png)
    



    
![png](../media/Lab01_files/Lab02_27_5.png)
    



Badając te kwestię postanowiłem sprawdzić najpierw testem Shapiro-Wilka, czy wartości są rozdystrybuowane w sposób normalny, jednak patrząc na histogramy danych i wyniki testu postanowiłem sprawdzić czy dane są w 'przybliżeniu' rozdystrybuowane w sposób normalny, więc postanowiłem sprawdzić mniej restrykcyjnym testem. Wykres kwantylowy (qqplot). Można z tego wyciągnąć, że atrybuty:
- thalach
- chol
- age

mają rozkład podobny do normalnego.

## 4. Dla cech kategorycznych: czy rozkład jest w przybliżeniu równomierny?


```python
categorical_variables = variable_info[(variable_info['type']=='Categorical')]['name']
categorical_variables
```




    1         sex
    2          cp
    5         fbs
    6     restecg
    8       exang
    10      slope
    12       thal
    Name: name, dtype: object




```python
categorical_df = X[categorical_variables]
categorical_df
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
      <th>sex</th>
      <th>cp</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>exang</th>
      <th>slope</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>3.0</td>
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
    </tr>
    <tr>
      <th>298</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 7 columns</p>
</div>




```python
for column in categorical_df.columns:
    plt.figure(figsize=(6, 6))
    categorical_df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Pie Chart for {column}')
    plt.ylabel('')
    plt.show()
```


    
![png](../media/Lab01_files/Lab02_32_0.png)
    



    
![png](../media/Lab01_files/Lab02_32_1.png)
    



    
![png](../media/Lab01_files/Lab02_32_2.png)
    



    
![png](../media/Lab01_files/Lab02_32_3.png)
    



    
![png](../media/Lab01_files/Lab02_32_4.png)
    



    
![png](../media/Lab01_files/Lab02_32_5.png)
    



    
![png](../media/Lab01_files/Lab02_32_6.png)
    


Rozkład wartości atrybutów kategorycznych jest różny, w większości przypadków nierównomierny, zależnie od atrybutu.
w niektórych przypadkach jest spora dysproporcja w danych, ale przeważnie dla jednego z 3 przypadków.
np. **thal,slope,restec**
W przypadku **sex, fbs** znacznie przeważa jedna kategoria.
Najbardziej równomierny jest zbiór **cp**


## 5. Czy występują cechy brakujące i jaką strategię możemy zastosować żeby je zastąpić?


```python
X.isnull().sum().sort_values(ascending=False)
```




    ca          4
    thal        2
    age         0
    sex         0
    cp          0
    trestbps    0
    chol        0
    fbs         0
    restecg     0
    thalach     0
    exang       0
    oldpeak     0
    slope       0
    dtype: int64




```python
y.isnull().sum()
```




    num    0
    dtype: int64



W przypadku **ca** uzupełnię brakujące wartości średnią. Ponieważ jest to wartość numeryczna. W przypadku **thal** uzupełnię je modą, ponieważ jest to atrybut kategoryczny.


```python
X
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>3</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>3.0</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>45</td>
      <td>1</td>
      <td>1</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>1.2</td>
      <td>2</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>68</td>
      <td>1</td>
      <td>4</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>0</td>
      <td>141</td>
      <td>0</td>
      <td>3.4</td>
      <td>2</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>57</td>
      <td>1</td>
      <td>4</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>2</td>
      <td>1.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>2</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>38</td>
      <td>1</td>
      <td>3</td>
      <td>138</td>
      <td>175</td>
      <td>0</td>
      <td>0</td>
      <td>173</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 13 columns</p>
</div>




```python
most_frequent_number_of_v = X['ca'].mode().iloc[0]
X.loc[:, 'ca'] = X['ca'].fillna(most_frequent_number_of_v)
most_frequent_category = X['thal'].mode().iloc[0]
X.loc[:, 'thal'] = X['thal'].fillna(most_frequent_category)
X.isnull().sum().sort_values(ascending=False)
```




    age         0
    sex         0
    cp          0
    trestbps    0
    chol        0
    fbs         0
    restecg     0
    thalach     0
    exang       0
    oldpeak     0
    slope       0
    ca          0
    thal        0
    dtype: int64



## 6. kod przekształcający dane do macierzy cech liczbowych (przykłady × cechy).


```python
df = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'],
                              prefix=['cp', 'restecg', 'slope', 'thal']).astype('int64')
df
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
      <th>age</th>
      <th>sex</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>ca</th>
      <th>cp_1</th>
      <th>...</th>
      <th>cp_4</th>
      <th>restecg_0</th>
      <th>restecg_1</th>
      <th>restecg_2</th>
      <th>slope_1</th>
      <th>slope_2</th>
      <th>slope_3</th>
      <th>thal_3.0</th>
      <th>thal_6.0</th>
      <th>thal_7.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>150</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>108</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>129</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <th>298</th>
      <td>45</td>
      <td>1</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>68</td>
      <td>1</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>141</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>300</th>
      <td>57</td>
      <td>1</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>115</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>0</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>38</td>
      <td>1</td>
      <td>138</td>
      <td>175</td>
      <td>0</td>
      <td>173</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 22 columns</p>
</div>



# Lab02 Prosta klasyfikacja



```python
X
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>3</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>3.0</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>45</td>
      <td>1</td>
      <td>1</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>1.2</td>
      <td>2</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>68</td>
      <td>1</td>
      <td>4</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>0</td>
      <td>141</td>
      <td>0</td>
      <td>3.4</td>
      <td>2</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>57</td>
      <td>1</td>
      <td>4</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>115</td>
      <td>1</td>
      <td>1.2</td>
      <td>2</td>
      <td>1.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>2</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>38</td>
      <td>1</td>
      <td>3</td>
      <td>138</td>
      <td>175</td>
      <td>0</td>
      <td>0</td>
      <td>173</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 13 columns</p>
</div>




```python
y = y.map(lambda x: 1 if x in (1,2,3,4) else 0)
y
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
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>1</td>
    </tr>
    <tr>
      <th>300</th>
      <td>1</td>
    </tr>
    <tr>
      <th>301</th>
      <td>1</td>
    </tr>
    <tr>
      <th>302</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 1 columns</p>
</div>



# Export przeanalizowanych i uzupelnionych plikow do csv


```python
from pathlib import Path
```


```python
csv_X_path = Path('../Dataset/X.csv')
csv_y_path = Path('../Dataset/y.csv')
```


```python
X.to_csv(csv_X_path,index=False)
y.to_csv(csv_y_path,index=False)
```

## Lab02
Sieci Neuronowe

Wynikiem implementacji listy jest program podzielony na 3 pliki:

- Functions.py -- funkcje pomocnicze, dyskretyzacja danych,
  normalizacja danych, obliczenie metryk dla zestawu testowego.

- Main.py -- środowisko testowe, w nim testuje wizualizuje
  zaimplementowane metody i funkcje.

- NeuralNetwork.py -- klasa prostej sieci neuronowej, opartej na
  gradiencie z entropii krzyżowej.

Implementacja rozwiązania:

Wyjście w sieci było implementowane na wzór:

𝑝(𝑥) = 𝜎(𝑊𝑥 + 𝑏)

``` Python
    def p(self, x):
        argument = np.dot(x, self.W) + self.b
        return self.sigmoid(argument)
```
gdzie funkcja sigmoid to:

<div style="text-align:center;">
 
  $\sigma(n) = \frac{1}{1 + e^{-n}}$
</div>

Co w pythonie może być osiągnięte za pomocą funkcji
[expit(x)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)
:

``` Python
    def sigmoid(self, n):
    return expit(n)
```
Jako funkcję kosztu wykorzystujemy entropię krzyżową:
<div style="text-align:center;">
    𝐿 = −𝑦 ln 𝑝(𝑥) − (1 − 𝑦) ln(1 − 𝑝(𝑥))
</div>

``` Python
    def cross_entropy_loss(self, y, y_pred):
        epsilon = 1e-15
        loss = y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred +epsilon)
        return -np.sum(loss)
```
epsilon został dodany ponieważ był problem z liczeniem wartości 0.

Implementacja gradientu będzie za pomocą pochodnej po wagach modelu, co
z entropii krzyżowej daje:
<div style="display: flex; justify-content: center; align-items: center; height: 200px;">
  
  $\frac{\partial L}{\partial w_i} = -(y - p(x))x$
  
  $\frac{\partial L}{\partial w_i} = (p(x) - y)x$
  
</div>

``` Python
    def compute_gradient(self, X_train, y_train):
        y_pred = self.p(X_train)
        dz = y_pred - y_train
        dw = np.dot(X_train.T, dz)
        db = np.sum(dz)
        return dw, db
```
Model uczy się na podstawie zmiany wag, tak aby iść w stronę wyznaczoną
przez gradient. Implementacja:

𝑤~i~′ = 𝑤~𝑖~ − 𝛼 $\frac{\partial L\ }{\partial wi}$

``` Python
    dw, db = self.compute_gradient(X_train, y_train)
```

Aktualizacja wag i bias zgodnie z gradientem i współczynnikiem
uczenia:

``` Python
    self.W -= self.learning_rate * dw
    self.b -= self.learning_rate * db
```
Mój model posiada również 3 funkcje które mogą go wyuczyć:

- Fit_model_covergence -- który mówi o wystarczająco małej zmianie aby
  przerwać proces uczenia.

- Fit -- podstawowa wersja ucząca model, przechodząca przez cały zbiór
  X razy.

- Fit_batches -- wersja rozszerzona o dzielenie zbioru na paczki o
  nadanej wielkości, przechodzi przez zbiór X razy.

Aby zmaksymalizować uczenie się modelu, dane przed każdą iteracją są
losowo mieszane.

Aby zobaczyć wpływ procesowania danych będę rozpatrywał wszystkie wyniki
kontekście 3 metod procesowania:

- Normalizacja

- Dyskretyzacja

- Surowe dane

Hiperparametry i parametry zostały wybrane dla każdego osobno tak aby
zmaksymalizować ich potencjał.

Wyniki uczenia dla parametrow i hiperparametrów:

**Surowe dane**

``` Python
    learning_rate_basic_without_b = 0.1
    learning_rate_basic_with_b = 0.001
    num_of_iterations_basic = 400
    batch_size = 100
```
| Obraz 1                            | Obraz 2                            | Obraz 3                            | Obraz 4                            |
|------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| ![Obraz 1](media/basic_data_1.png) | ![Obraz 2](media/basic_data_2.png) | ![Obraz 3](media/basic_data_3.png) | ![Obraz 4](media/basic_data_4.png) |
| Obraz 5                            | Obraz 6                            | Obraz 7                            | Obraz 8                            |
| ![Obraz 5](media/basic_data_5.png) | ![Obraz 6](media/basic_data_6.png) | ![Obraz 7](media/basic_data_7.png) | ![Obraz 8](media/basic_data_8.png) |

Możemy stąd zauważyć, że dane paczkowane, mają lepszy wynik ale są mniej
stabilne jeśli chodzi o metryki i proces uczenia.

Jendak oba wyniki są bardzo dobre, plasują się na poziomie \>=0.6 jeśli
chodzi o wszystkie metryki, co jest lepsze niż losowe zgadywanie.
Najlepsza

**Dane poddane dyskretyzacji:**

``` Python
learning_rate_discrete_without_b = 0.0005
learning_rate_discrete_with_b = 0.0005
batch_size = 64
num_of_iterations_discretization = 60
```
| Obraz 1                                 | Obraz 2                                 | Obraz 3                                 | Obraz 4                                 |
|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| ![Obraz 1](media/discretize_data_1.png) | ![Obraz 2](media/discretize_data_2.png) | ![Obraz 3](media/discretize_data_3.png) | ![Obraz 4](media/discretize_data_4.png) |
| Obraz 5                                 | Obraz 6                                 | Obraz 7                                 | Obraz 8                                 |
| ![Obraz 5](media/discretize_data_5.png) | ![Obraz 6](media/discretize_data_6.png) | ![Obraz 7](media/discretize_data_7.png) | ![Obraz 8](media/discretize_data_8.png) |

**Dane poddane normalizacji:**

``` Python
learning_rate_normalization_without_b = 0.001
learning_rate_normalization_with_b = 0.001
num_of_iterations_normalization = 200
batch_size= 128
```
| Obraz 1                                | Obraz 2                                | Obraz 3                                | Obraz 4                                |
|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| ![Obraz 1](media/normalize_data_1.png) | ![Obraz 2](media/normalize_data_2.png) | ![Obraz 3](media/normalize_data_3.png) | ![Obraz 4](media/normalize_data_4.png) |
| Obraz 5                                | Obraz 6                                | Obraz 7                                | Obraz 8                                |
| ![Obraz 5](media/normalize_data_5.png) | ![Obraz 6](media/normalize_data_6.png) | ![Obraz 7](media/normalize_data_7.png) | ![Obraz 8](media/normalize_data_8.png) |

W tym przypadku możemy zauważyć, że jest zdecydowanie mniej iteracji bo
tylko 60 i model się stabilizuje, w przypadku paczkowania pomimo braku
wyraźnej różnicy na wykresie kosztu możemy zauważyć różnicę w miarach.
Wszystkie testowe miary wskazują ok. 90% poprawności, co jest wspaniałym
wynikiem.

W przypadku paczkowania możemy także zauważyć mniejsze wahania metryk.

Wnioski:

Dobranie odpowiednich parametrów i hiperparametrów odgrywa kluczową rolę
w powodzeniu modelu. Jest to ciężkie bez znajomości metod na znalezienie
optmalnych współczynników, trzeba sprawdzać to metodą prób i błędów.

Preprocessing też odgrywa ważną rolę w tym jak sprawuje się model.

Nawet taki prosty model może sobie dobrze radzić z binarną klasyfikacją.

## Lab03
