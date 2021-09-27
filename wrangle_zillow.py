import os
import pandas as pd
import env
import numpy as np
from scipy import stats

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

import prepare
import acquire

def wrangle_zillow():
    # To get the single unit properties
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]

    # Importing the data into a pandas dataframe
    df = acquire.get_zillow_data()

    # Returns a summary of the dataframe
    acquire.summarize_df(df)

    # Get only single use properties
    df = df[df.propertylandusetypeid.isin(single_use)]

    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # removing bedroom and bathroom counts above 7
    df = df[df.bedroomcnt < 7]
    df = df[df.bathroomcnt < 7]
    
    # removing houses over 10000 square feet
    df = df[df.calculatedfinishedsquarefeet < 8000]

    # Removes columns with too high percentage of nulls
    df = acquire.handle_missing_values(df)

    # Get dummie variables for our counties
    df['county'] = df.fips.apply(lambda x: 'orange' if x == 6059.0 else 'los_angeles' if x == 6037.0 else 'ventura')
    df = df.drop(columns=['fips'])
    dummies = pd.get_dummies(df['county'])
    df = pd.concat([df, dummies],axis=1)

    # adding tax rate percentage
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # adding structure square footage cost
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet

    # adding land square footage cost
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    # removing outliers on lotsize
    df = df[df.lotsizesquarefeet < 200000]

    # removing columns that are of no use
    dropcols = ['parcelid',
                'calculatedbathnbr',
                'finishedsquarefeet12',
                'fullbathcnt',
                'heatingorsystemtypeid',
                'propertycountylandusecode',
                'propertylandusetypeid',
                'propertyzoningdesc',
                'censustractandblock',
                'propertylandusedesc',
                'unitcnt',
                'transactiondate',
                'county',
               'heatingorsystemdesc',
               'id',
               'assessmentyear']
    
    df.drop(columns = dropcols, inplace = True)

    # filling nulls
    # assume that since this is Southern CA, null means 'None' for heating system
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Renaming columns for readibility
    df.rename(columns = {"bathroomcnt":"bathrooms",
                        "bedroomcnt":"bedrooms",
                        "buildingqualitytypeid ":"building_quality",
                        "calculatedfinishedsquarefeet":"square_footage",
                        "lotsizesquarefeet":"lot_size",
                        "rawcensustractandblock":"census_tract_and_block",
                        "regionidcity":"city_id",
                        "regionidcounty":"county_id",
                        "regionidzip":"zip_id",
                        "roomcnt":"room_count",
                        "yearbuilt":"year_built",
                        "structuretaxvaluedollarcnt":"structure_tax_value",
                        "taxvaluedollarcnt":"tax_value",
                        "logerror":"target"}, inplace = True)
    # dropping the null values
    df = df.dropna()

    # removing outliers
    df = prepare.remove_outliers(df)

    # train validate test split
    train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = prepare.train_validate_test(df, target = "target")

    # scaling our data
    scaler, X_train_scaled, X_validate_scaled, X_test_scaled = prepare.min_max_scaler(X_train, X_validate, X_test)
    
    return df, scaler, train, validate, test, X_train, X_train_scaled, y_train, X_validate, X_validate_scaled, y_validate, X_test, X_test_scaled, y_test