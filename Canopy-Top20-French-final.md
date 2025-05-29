```python
pip install pandas numpy
```

    Requirement already satisfied: pandas in c:\anaconda\lib\site-packages (2.0.3)
    Requirement already satisfied: numpy in c:\users\ranju\appdata\roaming\python\python311\site-packages (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\anaconda\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\anaconda\lib\site-packages (from pandas) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in c:\anaconda\lib\site-packages (from pandas) (2023.3)
    Requirement already satisfied: six>=1.5 in c:\anaconda\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import pandas as pd
import numpy as np
```


```python
# Load the first dataset
df1 = pd.read_csv('Canopy-Dataset/Dataset-Movie-details.csv')

# Load the second dataset
df2 = pd.read_csv('Canopy-Dataset/Dataset-Movies-online-streaming-services.csv')

```


```python
# Display the first few rows of the first DataFrame
print(df1.head())
```

                                Title  Year  Age  IMDb Rotten Tomatoes  \
    0                       Inception  2010  13+   8.8             87%   
    1                      The Matrix  1999  18+   8.7             87%   
    2          Avengers: Infinity War  2018  13+   8.5             84%   
    3              Back to the Future  1985   7+   8.5             96%   
    4  The Good, the Bad and the Ugly  1966  18+   8.8             97%   
    
                            Directors                            Genres  \
    0               Christopher Nolan  Action,Adventure,Sci-Fi,Thriller   
    1  Lana Wachowski,Lilly Wachowski                     Action,Sci-Fi   
    2         Anthony Russo,Joe Russo           Action,Adventure,Sci-Fi   
    3                 Robert Zemeckis           Adventure,Comedy,Sci-Fi   
    4                    Sergio Leone                           Western   
    
                            Country                 Language  Runtime  
    0  United States,United Kingdom  English,Japanese,French    148.0  
    1                 United States                  English    136.0  
    2                 United States                  English    149.0  
    3                 United States                  English    116.0  
    4      Italy,Spain,West Germany                  Italian    161.0  
    


```python
# Display the first few rows of the second DataFrame
print(df2.head())
```

       ID  Title  Netflix  Hulu  Prime Video  Disney+
    0   1   1:54        0     0            1        0
    1   2   2:22        0     1            0        0
    2   3   3:19        0     0            1        0
    3   4   7:19        1     0            0        0
    4   5  11:11        0     0            1        0
    


```python
# Get summary statistics for the first DataFrame
print(df1.describe())

```

                   Year          IMDb       Runtime
    count  16744.000000  16173.000000  16152.000000
    mean    2003.014035      5.902751     93.413447
    std       20.674321      1.347867     28.219222
    min     1902.000000      0.000000      1.000000
    25%     2000.000000      5.100000     82.000000
    50%     2012.000000      6.100000     92.000000
    75%     2016.000000      6.900000    104.000000
    max     2020.000000      9.300000   1256.000000
    


```python
# Get summary statistics for the second DataFrame
print(df2.describe())
```

                     ID       Netflix          Hulu   Prime Video       Disney+
    count  16744.000000  16744.000000  16744.000000  16744.000000  16744.000000
    mean    8372.500000      0.212613      0.053930      0.737817      0.033684
    std     4833.720789      0.409169      0.225886      0.439835      0.180419
    min        1.000000      0.000000      0.000000      0.000000      0.000000
    25%     4186.750000      0.000000      0.000000      0.000000      0.000000
    50%     8372.500000      0.000000      0.000000      1.000000      0.000000
    75%    12558.250000      0.000000      0.000000      1.000000      0.000000
    max    16744.000000      1.000000      1.000000      1.000000      1.000000
    


```python
# Check for missing values in the first DataFrame
print(df1.isnull().sum())
```

    Title                  0
    Year                   0
    Age                 9390
    IMDb                 571
    Rotten Tomatoes    11586
    Directors            726
    Genres               275
    Country              435
    Language             614
    Runtime              592
    dtype: int64
    


```python
# Check for missing values in the second DataFrame
print(df2.isnull().sum())
```

    ID             0
    Title          0
    Netflix        0
    Hulu           0
    Prime Video    0
    Disney+        0
    dtype: int64
    


```python
# Get the shape of the first DataFrame
print(df1.shape)

# Get the data types of columns in the first DataFrame
print(df1.dtypes)

```

    (16744, 10)
    Title               object
    Year                 int64
    Age                 object
    IMDb               float64
    Rotten Tomatoes     object
    Directors           object
    Genres              object
    Country             object
    Language            object
    Runtime            float64
    dtype: object
    


```python
# Get the shape of the second DataFrame
print(df2.shape)

# Get the data types of columns in the second DataFrame
print(df2.dtypes)

```

    (16744, 6)
    ID              int64
    Title          object
    Netflix         int64
    Hulu            int64
    Prime Video     int64
    Disney+         int64
    dtype: object
    


```python
import pandas as pd
import numpy as np

# Load the datasets
df1 = pd.read_csv('Canopy-Dataset/Dataset-Movie-details.csv')
df2 = pd.read_csv('Canopy-Dataset/Dataset-Movies-online-streaming-services.csv')

# Combine the datasets
combined_df = pd.merge(df1, df2, on='Title', how='outer')

```


```python
combined_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16895 entries, 0 to 16894
    Data columns (total 15 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Title            16895 non-null  object 
     1   Year             16744 non-null  float64
     2   Age              7354 non-null   object 
     3   IMDb             16173 non-null  float64
     4   Rotten Tomatoes  5158 non-null   object 
     5   Directors        16018 non-null  object 
     6   Genres           16469 non-null  object 
     7   Country          16309 non-null  object 
     8   Language         16130 non-null  object 
     9   Runtime          16152 non-null  float64
     10  ID               16744 non-null  float64
     11  Netflix          16744 non-null  float64
     12  Hulu             16744 non-null  float64
     13  Prime Video      16744 non-null  float64
     14  Disney+          16744 non-null  float64
    dtypes: float64(8), object(7)
    memory usage: 1.9+ MB
    


```python
combined_df.columns
```




    Index(['Title', 'Year', 'Age', 'IMDb', 'Rotten Tomatoes', 'Directors',
           'Genres', 'Country', 'Language', 'Runtime', 'ID', 'Netflix', 'Hulu',
           'Prime Video', 'Disney+'],
          dtype='object')




```python
# List of columns to drop
columns_to_drop = ['Rotten Tomatoes','Runtime', 'ID']

# Drop the specified columns
combined_df = combined_df.drop(columns=columns_to_drop, axis=1)
```


```python
# Handle missing values
for column in combined_df.select_dtypes(include=[np.number]).columns:
    combined_df[column].fillna(combined_df[column].mean(), inplace=True)

for column in combined_df.select_dtypes(include=[object]).columns:
    combined_df[column].fillna(combined_df[column].mode()[0], inplace=True)
```


```python
# Filter for French-language movies original & dubbed
french_movies = combined_df[combined_df['Language'].str.contains('French', case=False, na=False)]

# Display the filtered DataFrame
print(french_movies)
```

                                            Title    Year  Age  IMDb  \
    0                                   Inception  2010.0  13+   8.8   
    7                            Django Unchained  2012.0  18+   8.4   
    9                        Inglourious Basterds  2009.0  18+   8.3   
    14            Monty Python and the Holy Grail  1975.0   7+   8.2   
    17                              Groundhog Day  1993.0   7+   8.0   
    ...                                       ...     ...  ...   ...   
    16560  The Brave Little Toaster to the Rescue  1997.0  all   6.3   
    16621                            The Other Me  2000.0  all   6.1   
    16656                    Diving with Dolphins  2020.0  18+   7.5   
    16666              Penguins: Life on the Edge  2020.0  18+   6.9   
    16677                           That Darn Cat  1997.0   7+   4.7   
    
                                      Directors  \
    0                         Christopher Nolan   
    7                         Quentin Tarantino   
    9                         Quentin Tarantino   
    14                Terry Gilliam,Terry Jones   
    17                             Harold Ramis   
    ...                                     ...   
    16560  Robert C. Ramirez,Patrick A. Ventura   
    16621                    Sotiris Tsafoulias   
    16656                         Keith Scholey   
    16666       Alastair Fothergill,Jeff Wilson   
    16677                      Robert Stevenson   
    
                                       Genres                       Country  \
    0        Action,Adventure,Sci-Fi,Thriller  United States,United Kingdom   
    7                           Drama,Western                 United States   
    9                     Adventure,Drama,War         Germany,United States   
    14               Adventure,Comedy,Fantasy                United Kingdom   
    17                 Comedy,Fantasy,Romance                 United States   
    ...                                   ...                           ...   
    16560  Animation,Adventure,Family,Fantasy                 United States   
    16621        Crime,Drama,Mystery,Thriller                        Greece   
    16656                         Documentary                 United States   
    16666                  Documentary,Family                 United States   
    16677        Comedy,Crime,Family,Thriller                 United States   
    
                                                  Language  Netflix  Hulu  \
    0                              English,Japanese,French      1.0   0.0   
    7                        English,German,French,Italian      1.0   0.0   
    9                        English,German,French,Italian      1.0   0.0   
    14                                English,French,Latin      1.0   0.0   
    17                              English,French,Italian      1.0   0.0   
    ...                                                ...      ...   ...   
    16560  Chinese,Mandarin,Japanese,Korean,French,English      0.0   0.0   
    16621                                     Greek,French      0.0   0.0   
    16656                                   English,French      0.0   0.0   
    16666                                   English,French      0.0   0.0   
    16677                                   English,French      0.0   0.0   
    
           Prime Video  Disney+  
    0              0.0      0.0  
    7              0.0      0.0  
    9              0.0      0.0  
    14             0.0      0.0  
    17             0.0      0.0  
    ...            ...      ...  
    16560          0.0      1.0  
    16621          0.0      1.0  
    16656          0.0      1.0  
    16666          0.0      1.0  
    16677          0.0      1.0  
    
    [800 rows x 12 columns]
    


```python
# Find the top 20 movies in each genre
top_20_movies = french_movies.groupby('Genres').apply(lambda x: x.nlargest(20, 'IMDb')).reset_index(drop=True)
top_20_movies
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
      <th>Title</th>
      <th>Year</th>
      <th>Age</th>
      <th>IMDb</th>
      <th>Directors</th>
      <th>Genres</th>
      <th>Country</th>
      <th>Language</th>
      <th>Netflix</th>
      <th>Hulu</th>
      <th>Prime Video</th>
      <th>Disney+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Black List</td>
      <td>1972.0</td>
      <td>18+</td>
      <td>5.6</td>
      <td>Alain Bonnot</td>
      <td>Action</td>
      <td>France</td>
      <td>French</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Bittersweet</td>
      <td>2017.0</td>
      <td>18+</td>
      <td>5.3</td>
      <td>D. Ho</td>
      <td>Action</td>
      <td>China</td>
      <td>Mandarin,Portuguese,Spanish,French,German</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Samouraïs</td>
      <td>2002.0</td>
      <td>18+</td>
      <td>3.3</td>
      <td>Giordano Gederlini</td>
      <td>Action</td>
      <td>Spain,France,Germany</td>
      <td>French,Japanese</td>
      <td>0.212613</td>
      <td>0.05393</td>
      <td>0.737817</td>
      <td>0.033684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8 Assassins</td>
      <td>2014.0</td>
      <td>18+</td>
      <td>4.7</td>
      <td>Said C. Naciri</td>
      <td>Action,Adventure</td>
      <td>Morocco</td>
      <td>Arabic,French</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Admiral</td>
      <td>2015.0</td>
      <td>7+</td>
      <td>7.0</td>
      <td>Roel Reiné</td>
      <td>Action,Adventure,Biography,Drama,History</td>
      <td>Netherlands</td>
      <td>Dutch,English,French</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>660</th>
      <td>Rebirth</td>
      <td>2016.0</td>
      <td>18+</td>
      <td>5.0</td>
      <td>Karl Mueller</td>
      <td>Thriller</td>
      <td>United States</td>
      <td>English,French</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>661</th>
      <td>Sex Doll</td>
      <td>2016.0</td>
      <td>18+</td>
      <td>4.5</td>
      <td>Sylvie Verheyde</td>
      <td>Thriller</td>
      <td>United Kingdom,France</td>
      <td>English,French</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>662</th>
      <td>The Statement</td>
      <td>2003.0</td>
      <td>18+</td>
      <td>6.2</td>
      <td>Norman Jewison</td>
      <td>Thriller,Drama</td>
      <td>Canada,France,United Kingdom</td>
      <td>English,German,Italian,Latin,French</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>663</th>
      <td>The Bridesmaid</td>
      <td>2004.0</td>
      <td>18+</td>
      <td>6.6</td>
      <td>Claude Chabrol</td>
      <td>Thriller,Drama,Romance</td>
      <td>France,Germany,Italy</td>
      <td>French</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>664</th>
      <td>Cemetery Without Crosses</td>
      <td>1969.0</td>
      <td>18+</td>
      <td>6.9</td>
      <td>Robert Hossein</td>
      <td>Western</td>
      <td>France,Italy</td>
      <td>French,Italian,Spanish</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>665 rows × 12 columns</p>
</div>




```python
# Define the mapping of main categories to sub-categories
# Add more categories as needed
genre_mapping = {
    'Action & Adventure': ['Action','Adventure','Thriller','Horror','Mystery','Crime'],
    'Documentry & Short': ['Documentry','Short','Biography'],
    'Fiction':['Animation','Fantasy','Comedy'],
    'Drama':['Drama','Family','Romance'],
    'Others':['Western','Music','Sci-Fi'],
         }
```


```python
# Function to map sub-categories to main categories based on prefix
def map_genre(genres, mapping):
    for main_category, prefixes in mapping.items():
        for prefix in prefixes:
            if genres.startswith(prefix):
                return main_category
    #return 'Others'  # Default category if no match is found
```


```python
# Apply the mapping function to create a new column
combined_df['Main_Genre'] = combined_df['Genres'].apply(lambda x: map_genre(x, genre_mapping))
```


```python
# Filter movies where the 'Language' column contains 'French'
french_movies = combined_df[combined_df['Language'].str.contains('French', case=False, na=False)]
```


```python
# Group by main categories and get the top 20 movies by IMDb rating
top_20_movies = french_movies.groupby('Main_Genre').apply(lambda x: x.nlargest(20, 'IMDb')).reset_index(drop=True)

# Display the filtered DataFrame
print(top_20_movies)
```

                                  Title    Year  Age  IMDb  \
    0                         Inception  2010.0  13+   8.8   
    1                    The Green Mile  1999.0  18+   8.6   
    2              Inglourious Basterds  2009.0  18+   8.3   
    3   Monty Python and the Holy Grail  1975.0   7+   8.2   
    4                             Queen  2013.0  13+   8.2   
    ..                              ...     ...  ...   ...   
    79               Cyrano de Bergerac  1950.0  18+   7.5   
    80         Cemetery Without Crosses  1969.0  18+   6.9   
    81                             2:22  2017.0  13+   5.8   
    82                 Blank Generation  1980.0  18+   4.9   
    83                             3022  2019.0  18+   4.5   
    
                        Directors                            Genres  \
    0           Christopher Nolan  Action,Adventure,Sci-Fi,Thriller   
    1              Frank Darabont       Crime,Drama,Fantasy,Mystery   
    2           Quentin Tarantino               Adventure,Drama,War   
    3   Terry Gilliam,Terry Jones          Adventure,Comedy,Fantasy   
    4                  Vikas Bahl            Adventure,Comedy,Drama   
    ..                        ...                               ...   
    79        Jean-Paul Rappeneau      Comedy,Drama,History,Romance   
    80             Robert Hossein                           Western   
    81                 John Suits                            Sci-Fi   
    82                Ulli Lommel                       Music,Drama   
    83                 John Suits                            Sci-Fi   
    
                             Country                             Language  \
    0   United States,United Kingdom              English,Japanese,French   
    1                  United States                       English,French   
    2          Germany,United States        English,German,French,Italian   
    3                 United Kingdom                 English,French,Latin   
    4                          India  Hindi,English,French,Japanese,Dutch   
    ..                           ...                                  ...   
    79                        France                               French   
    80                  France,Italy               French,Italian,Spanish   
    81                 United States                       English,French   
    82                 United States                       English,French   
    83                 United States                       English,French   
    
        Netflix  Hulu  Prime Video  Disney+          Main_Genre  
    0       1.0   0.0          0.0      0.0  Action & Adventure  
    1       0.0   1.0          0.0      0.0  Action & Adventure  
    2       1.0   0.0          0.0      0.0  Action & Adventure  
    3       1.0   0.0          0.0      0.0  Action & Adventure  
    4       0.0   0.0          1.0      0.0  Action & Adventure  
    ..      ...   ...          ...      ...                 ...  
    79      0.0   0.0          1.0      0.0             Fiction  
    80      0.0   0.0          1.0      0.0              Others  
    81      0.0   1.0          0.0      0.0              Others  
    82      0.0   0.0          1.0      0.0              Others  
    83      1.0   0.0          0.0      0.0              Others  
    
    [84 rows x 13 columns]
    


```python
print("\nGenre Distribution in French-language Movies:")
print(genres_distribution)
```

    
    Genre Distribution in French-language Movies:
               Main_Genre  count
    0               Drama    238
    1  Action & Adventure    217
    2             Fiction    182
    3  Documentry & Short     47
    4              Others      4
    


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Genres distribution plot
plt.figure(figsize=(23, 7))
sns.countplot(data=french_movies, x='Main_Genre', order=french_movies['Main_Genre'].value_counts().index)
plt.title('Genre Distribution of French-language Movies')
plt.xlabel('Main_Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=40)
plt.show()
```


    
![png](output_25_0.png)
    



```python
# Top 20 movies in each genre plot
plt.figure(figsize=(12, 24))
for genres in top_20_movies['Main_Genre'].unique():
    genres_data = top_20_movies[top_20_movies['Main_Genre'] == genres]
    sns.scatterplot(data=genres_data, x='IMDb', y='Title', label=genres)
plt.title('Top 20 Movies in Each Genre (French-language)')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.legend()
plt.show()
```


    
![png](output_26_0.png)
    



```python
# Ratings distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(french_movies['Main_Genre'], bins=20, kde=True)
plt.title('Distribution of Main Genre French-language Movies')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
```




    Text(0, 0.5, 'Frequency')




    
![png](output_27_1.png)
    



```python
top_20_movies = french_movies.nlargest(20,'IMDb')[['Title','IMDb','Directors','Genres']].set_index('Title')
top_20_movies
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
      <th>IMDb</th>
      <th>Directors</th>
      <th>Genres</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Inception</th>
      <td>8.8</td>
      <td>Christopher Nolan</td>
      <td>Action,Adventure,Sci-Fi,Thriller</td>
    </tr>
    <tr>
      <th>The Green Mile</th>
      <td>8.6</td>
      <td>Frank Darabont</td>
      <td>Crime,Drama,Fantasy,Mystery</td>
    </tr>
    <tr>
      <th>It's a Wonderful Life</th>
      <td>8.6</td>
      <td>Frank Capra</td>
      <td>Drama,Family,Fantasy</td>
    </tr>
    <tr>
      <th>Home</th>
      <td>8.6</td>
      <td>Tim Johnson</td>
      <td>Animation,Adventure,Comedy,Family,Fantasy,Sci-Fi</td>
    </tr>
    <tr>
      <th>Inner Worlds, Outer Worlds</th>
      <td>8.6</td>
      <td>Daniel Schmidt</td>
      <td>Documentary,History</td>
    </tr>
    <tr>
      <th>Senna</th>
      <td>8.5</td>
      <td>Asif Kapadia</td>
      <td>Documentary,Biography,Sport</td>
    </tr>
    <tr>
      <th>Beyond the Summits</th>
      <td>8.5</td>
      <td>Rémy Tezier</td>
      <td>Documentary</td>
    </tr>
    <tr>
      <th>The Lion King</th>
      <td>8.5</td>
      <td>Jon Favreau</td>
      <td>Animation,Adventure,Drama,Family,Musical</td>
    </tr>
    <tr>
      <th>Django Unchained</th>
      <td>8.4</td>
      <td>Quentin Tarantino</td>
      <td>Drama,Western</td>
    </tr>
    <tr>
      <th>Inglourious Basterds</th>
      <td>8.3</td>
      <td>Quentin Tarantino</td>
      <td>Adventure,Drama,War</td>
    </tr>
    <tr>
      <th>The Weekend Sailor</th>
      <td>8.3</td>
      <td>Bernardo Arsuaga</td>
      <td>Documentary</td>
    </tr>
    <tr>
      <th>Calamity Jane: Légende de l'Ouest</th>
      <td>8.3</td>
      <td>Gregory Monro</td>
      <td>Documentary,Biography,Drama,Western</td>
    </tr>
    <tr>
      <th>Coexist</th>
      <td>8.3</td>
      <td>Adam Mazo</td>
      <td>Documentary,History,News,War</td>
    </tr>
    <tr>
      <th>Monty Python and the Holy Grail</th>
      <td>8.2</td>
      <td>Terry Gilliam,Terry Jones</td>
      <td>Adventure,Comedy,Fantasy</td>
    </tr>
    <tr>
      <th>Virunga</th>
      <td>8.2</td>
      <td>Orlando von Einsiedel</td>
      <td>Documentary,War</td>
    </tr>
    <tr>
      <th>Skydancers</th>
      <td>8.2</td>
      <td>Fredric Lean</td>
      <td>Documentary</td>
    </tr>
    <tr>
      <th>Portrait of a Lady on Fire</th>
      <td>8.2</td>
      <td>Céline Sciamma</td>
      <td>Drama,Romance</td>
    </tr>
    <tr>
      <th>Downfall</th>
      <td>8.2</td>
      <td>Oliver Hirschbiegel</td>
      <td>Biography,Drama,History,War</td>
    </tr>
    <tr>
      <th>A Trip to the Moon</th>
      <td>8.2</td>
      <td>Georges Méliès</td>
      <td>Short,Action,Adventure,Comedy,Fantasy,Sci-Fi</td>
    </tr>
    <tr>
      <th>Queen</th>
      <td>8.2</td>
      <td>Vikas Bahl</td>
      <td>Adventure,Comedy,Drama</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Verify that the DataFrame contains the columns for OTT service providers
ott_columns = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
for col in ott_columns:
    if col not in combined_df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
```


```python
# Filter to only include French movies
french_movies = combined_df[combined_df['Language'].str.contains('French', na=False)]
```


```python
# Function to get top 20 movies for a given OTT service
def get_top_20_movies_by_service(df, service_column):
    return df[df[service_column] == 1].nlargest(20, 'IMDb')
```


```python
# Get top 20 French movies for each service provider
top_20_netflix = get_top_20_movies_by_service(french_movies, 'Netflix')
top_20_hulu = get_top_20_movies_by_service(french_movies, 'Hulu')
top_20_prime_video = get_top_20_movies_by_service(french_movies, 'Prime Video')
top_20_disney_plus = get_top_20_movies_by_service(french_movies, 'Disney+')
```


```python
# Combine all results into a single DataFrame
top_20_movies = pd.concat([top_20_netflix, top_20_hulu, top_20_prime_video, top_20_disney_plus]).reset_index(drop=True)

# Display the combined DataFrame
print(top_20_movies)
```

                                       Title    Year  Age  IMDb  \
    0                              Inception  2010.0  13+   8.8   
    1                                  Senna  2010.0  13+   8.5   
    2                       Django Unchained  2012.0  18+   8.4   
    3                   Inglourious Basterds  2009.0  18+   8.3   
    4        Monty Python and the Holy Grail  1975.0   7+   8.2   
    ..                                   ...     ...  ...   ...   
    75                      Tuck Everlasting  2002.0   7+   6.6   
    76                       The Parent Trap  1998.0   7+   6.5   
    77    National Treasure: Book of Secrets  2007.0   7+   6.5   
    78                                 Dumbo  2019.0   7+   6.3   
    79  Lilo & Stitch 2: Stitch Has a Glitch  2005.0   7+   6.3   
    
                          Directors                                    Genres  \
    0             Christopher Nolan          Action,Adventure,Sci-Fi,Thriller   
    1                  Asif Kapadia               Documentary,Biography,Sport   
    2             Quentin Tarantino                             Drama,Western   
    3             Quentin Tarantino                       Adventure,Drama,War   
    4     Terry Gilliam,Terry Jones                  Adventure,Comedy,Fantasy   
    ..                          ...                                       ...   
    75                  Jay Russell              Drama,Family,Fantasy,Romance   
    76                 Nancy Meyers     Adventure,Comedy,Drama,Family,Romance   
    77               Jon Turteltaub  Action,Adventure,Family,Mystery,Thriller   
    78                   Tim Burton                  Adventure,Family,Fantasy   
    79  Michael LaBash,Tony Leondis      Animation,Comedy,Drama,Family,Sci-Fi   
    
                                              Country  \
    0                    United States,United Kingdom   
    1             United Kingdom,France,United States   
    2                                   United States   
    3                           Germany,United States   
    4                                  United Kingdom   
    ..                                            ...   
    75                                  United States   
    76                   United States,United Kingdom   
    77                                  United States   
    78  United States,United Kingdom,Australia,Canada   
    79                                  United States   
    
                                  Language  Netflix  Hulu  Prime Video  Disney+  \
    0              English,Japanese,French      1.0   0.0          0.0      0.0   
    1   English,Portuguese,French,Japanese      1.0   0.0          0.0      0.0   
    2        English,German,French,Italian      1.0   0.0          0.0      0.0   
    3        English,German,French,Italian      1.0   0.0          0.0      0.0   
    4                 English,French,Latin      1.0   0.0          0.0      0.0   
    ..                                 ...      ...   ...          ...      ...   
    75                      English,French      0.0   0.0          0.0      1.0   
    76                      English,French      0.0   0.0          0.0      1.0   
    77                      English,French      0.0   0.0          0.0      1.0   
    78                      English,French      0.0   0.0          0.0      1.0   
    79              English,French,Spanish      0.0   0.0          0.0      1.0   
    
                Main_Genre  
    0   Action & Adventure  
    1                 None  
    2                Drama  
    3   Action & Adventure  
    4   Action & Adventure  
    ..                 ...  
    75               Drama  
    76  Action & Adventure  
    77  Action & Adventure  
    78  Action & Adventure  
    79             Fiction  
    
    [80 rows x 13 columns]
    


```python
# 20 top rated French movies & service providers
selected_columns = ['IMDb'] + ott_columns
top_20_movies = top_20_movies[selected_columns]
```


```python
# Display the result
print(top_20_movies)
```

           IMDb   Netflix     Hulu  Prime Video   Disney+
    0       8.8  1.000000  0.00000     0.000000  0.000000
    3564    8.6  0.000000  1.00000     0.000000  0.000000
    4439    8.6  0.000000  0.00000     1.000000  0.000000
    5063    8.6  0.000000  0.00000     1.000000  0.000000
    6843    8.6  0.000000  0.00000     1.000000  0.000000
    47      8.5  1.000000  0.00000     0.000000  0.000000
    9348    8.5  0.000000  0.00000     1.000000  0.000000
    16214   8.5  0.000000  0.00000     0.000000  1.000000
    7       8.4  1.000000  0.00000     0.000000  0.000000
    9       8.3  1.000000  0.00000     0.000000  0.000000
    8014    8.3  0.000000  0.00000     1.000000  0.000000
    8335    8.3  0.212613  0.05393     0.737817  0.033684
    11670   8.3  0.000000  0.00000     1.000000  0.000000
    14      8.2  1.000000  0.00000     0.000000  0.000000
    164     8.2  1.000000  0.00000     0.000000  0.000000
    1395    8.2  1.000000  0.00000     0.000000  0.000000
    3577    8.2  0.000000  1.00000     0.000000  0.000000
    4440    8.2  0.000000  0.00000     1.000000  0.000000
    4467    8.2  0.000000  0.00000     1.000000  0.000000
    4506    8.2  0.000000  0.00000     1.000000  0.000000
    
