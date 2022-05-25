from categorical_features import categorical_vars, all_date_vars, numeric_vars
import subprocess, time, string
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler

from typing import Dict, List

print ("TIME BINNING AND NORMALIZATION...")

import warnings
warnings.simplefilter(action='ignore')

pipInstall = "pip install pandas==1.3.1" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np

# testdate = pd.date_range(start=ca_created_at.min(), end=ca_created_at.max(), freq=dayz+'D', closed='right')[0]

def create_repeat(df2: pd.DataFrame, features, ca_created_at, batch_size='90') -> Dict[str,pd.DataFrame]:
    
    ca_created_at_ = pd.DataFrame(ca_created_at.index, columns=['ca_id'], index=ca_created_at.values)
    
    categorical_vars_ = set(features).intersection(categorical_vars)
    all_date_vars_ = set(features).intersection(all_date_vars)
    numeric_vars_ = set(features).intersection(numeric_vars)
        
    def numeric_handling(df: pd.DataFrame) -> pd.DataFrame:
    
        if len(numeric_vars_) > 0:
            for numeric_col in numeric_vars_:
                null_values = df.loc[:, numeric_col].isnull()
                df[numeric_col].fillna(-1, inplace = True)
                df.loc[~null_values, [numeric_col]] = StandardScaler().fit_transform(df.loc[~null_values, [numeric_col]]).astype(float)

        return df
    
    def categorical_handling(df: pd.DataFrame) -> pd.DataFrame:

        if len(categorical_vars_) > 0:
            df = pd.get_dummies(df, columns=categorical_vars_, drop_first=True)

        dummy_cols = df.select_dtypes(['uint8', 'bool']).columns

        def sum_product_division(a,b):
            if b!=0: 
                return a/b
            else:
                return 0

        '''
        application of FAMD algorithm applied for both categorical and numerical variables below
        '''

        df.loc[:, dummy_cols] = (df.loc[:, dummy_cols]
                                         .applymap(lambda x: sum_product_division(x, (np.sum(x)/df.shape[0])**.5) - (np.sum(x)/df.shape[0])))       

        print ("Z-Scored and Feature-Engineered DataFrame:\n", df.shape)

        return df
    
    def find_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
        
        constant_cols = [col for col in df.columns if df[col].std() < .01]
        if len(constant_cols) > 0:
            print ('quasi-constant columns to be removed:', constant_cols)
            df.drop(constant_cols, axis=1, inplace=True)
            
        return df

    out = defaultdict(dict)
    
    print ('NORMALIZE AND STANDARDIZE TIME SEGMENTS...\n')

    for dayz in list(map(str, [batch_size])):

        for ix, d in enumerate(pd.date_range(start=ca_created_at.min(), end=ca_created_at.max(), freq=dayz+'D', closed='right')):

            print ("time segment:", ix + 1, dayz, 'day window ending', d.date())

            if ix == 0:
                normalized = categorical_handling(numeric_handling(df2.iloc[np.where( np.in1d(df2.index, ca_created_at_[ca_created_at.min():d]))]))
                out[dayz][d.date()] = find_constant_cols(normalized)
                last = d
            else:
                normalized = categorical_handling(numeric_handling(df2.iloc[np.where( np.in1d(df2.index, ca_created_at_[last:d]))]))
                out[dayz][d.date()] = find_constant_cols(normalized)
                last = d
    return out
    
    
    