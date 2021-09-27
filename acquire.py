import os
import pandas as pd
import env
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    Returns zillow into a dataframe
    '''
    sql_query = '''
SELECT *
FROM properties_2017
JOIN (SELECT parcelid, logerror, MAX(transactiondate) as transactiondate 
FROM predictions_2017 group by parcelid, logerror) as pred_2017 using(parcelid) 
LEFT JOIN airconditioningtype using(airconditioningtypeid)
LEFT JOIN architecturalstyletype using(architecturalstyletypeid)
LEFT JOIN buildingclasstype using(buildingclasstypeid)
LEFT JOIN heatingorsystemtype using(heatingorsystemtypeid)
LEFT JOIN propertylandusetype using(propertylandusetypeid)
LEFT JOIN storytype using(storytypeid)
LEFT JOIN typeconstructiontype using(typeconstructiontypeid)
WHERE properties_2017.latitude is not null
and properties_2017.longitude is not null;
'''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.set_index('parcelid')
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    return df 

def get_zillow_data():
    '''get connection, returns zillow into a dataframe and creates a csv for us'''
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        df = new_zillow_data()
        df.to_csv('zillow.csv')
        
    return df

def summarize_df(df):
    print('-----Head-------')
    print(df.head(3))
    print('-----shape------')
    print('{} rows and {} columns'.format(df.shape[0], df.shape[1]))
    print('---info---')
    print(df.info())
    print(df.describe())
    print('----Catagorical Variables----')
    print(df.select_dtypes(include='object').columns.tolist())
    print('----Continous  Variables----')
    print(df.select_dtypes(exclude='object').columns.tolist())
    print('--nulls--')
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('=====================================================')
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    print(df.isna().sum())

def handle_outliers(df, col):
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3-q1 #Interquartile range
    lower_bound  = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > df[col].max():
        upper_bound = df[col].max()
    df_out = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df_out


def nulls_by_col(df):
    num_missing = df.isnull().sum()
    print(type(num_missing))
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing
    
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
	#function that will drop rows or columns based on the percent of values that are missing:\
	#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing

def remove_columns(df, cols_to_remove):
	#remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df