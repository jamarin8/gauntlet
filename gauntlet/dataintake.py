from categorical_features import categorical_vars, all_date_vars, numeric_vars
import subprocess, time, string, pickle
from collections import defaultdict, Counter

from time_bin_normalize import create_repeat
from params import params

print ("ENGAGING DATA INTAKE & PROCESSING MODULE...")
tic = time.perf_counter()

pipInstall = "pip install pandas" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
import pandas as pd
import numpy as np

pipInstall = "pip install smart_open" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
from smart_open import smart_open

from urllib.parse import unquote
def parse_url_seps(string: str) -> str:
    try:
        value = unquote(string)
    except TypeError:
        value = np.nan
    return value

pd.options.mode.chained_assignment = None
dependant_variable = 's3://sagemaker-shared-resources/ds_avant_data/unsupervised_learning/avant_unsupervised_dvs_01_18_2022.csv'
df_dvs = pd.read_csv(smart_open(dependant_variable), low_memory = False)
df_dvs['ca_id'] = df_dvs['ca_id'].astype('int')
df_dvs['customer_id'] = df_dvs['customer_id'].astype('int')
assert 'any_fraud' in df_dvs.columns

giact = 's3://sagemaker-shared-resources/ds_avant_data/unsupervised_learning/ds_avant_unsupervised_ts_2022_03_21.csv'
df_03_21 = pd.read_csv(smart_open(giact), low_memory = False)
df_03_21['ca_id'] = df_03_21['ca_id'].astype('int')
df_03_21['customer_id'] = df_03_21['customer_id'].astype('int')

df1 = df_dvs.merge(df_03_21, how='inner', on=['ca_id', 'customer_id'])

if len(df1.filter(like='ca_created').columns) > 1:
    df1 = df1.drop('ca_created_at_x', axis=1)
    df1['ca_created_at'] = df1['ca_created_at_y']
    df1 = df1.drop('ca_created_at_y', axis=1)
    
fraud_scores = 's3://sagemaker-shared-resources/ds_avant_data/unsupervised_learning/avant_unsupervised_fm_scoreset_01_18_2022.csv'
fm5 = pd.read_csv(smart_open(fraud_scores), low_memory = False)
fm5['ca_id'] = fm5['ca_id'].astype('int')
fm5['customer_id'] = fm5['customer_id'].astype('int')

df1_fm5 = df1.merge(fm5, how='inner', on=['ca_id', 'customer_id'])
df2 = df1_fm5.copy()
df2 = df2.query('product_type == "installment"')

print ('\t>> shape of merged intake dataframe', df2.shape)

df2.loc[:,'time_spent_on_rates_terms'] = df2.loc[:, 'time_spent_on_rates_terms'].apply(pd.to_timedelta).fillna(pd.Timedelta(seconds=0))
df2.loc[:,'time_spent_on_rates_terms'] = df2.time_spent_on_rates_terms.dt.total_seconds()

df2['gr_verification_response'] = df2['gr_verification_response'].fillna("MISSING")
df2['gr_account_response_code'] = df2['gr_account_response_code'].fillna("MISSING")
df2['gr_customer_response_code'] = df2['gr_customer_response_code'].fillna("MISSING")

# BUCKETIZE FSL SOURCE
df2['fsl_source'] = df2['fsl_source'].str.replace('google.*', 'google', regex=True)
df2['fsl_source'] = np.where(np.in1d(df2['fsl_source'], ['google', 'organic']), df2['fsl_source'], 'OTHER')

# INDEX CREATION
df2['ca_created_at'] = pd.to_datetime(df2['ca_created_at'])
df2 = df2.set_index('ca_id').sort_values(by=['ca_created_at'])

df2 = df2[~df2.fm_5_score.isna()]
fm_5 = df2.fm_5_score
with open('fm_5.pkl', 'wb') as f:
    pickle.dump(fm_5, f)
    
ca_created_at = df2.ca_created_at
with open('ca_created_at.pkl', 'wb') as f:
    pickle.dump(ca_created_at, f)
    
print ('done with ca_created')
    
def fraud_selection(df, type_of_fraud):
        
    if type_of_fraud == 'general':
        df['fraud_specified'] = ((df['dep_var']==1) | (df['first_party_fraud']==1)).astype(int).to_list()
    elif type_of_fraud == 'first':
        df['fraud_specified'] = (df['first_party_fraud']==1).astype(int).to_list()
    elif type_of_fraud == 'third':
        df['fraud_specified'] = (df['dep_var']==1).astype(int).to_list()
    elif type_of_fraud == 'any':
        df['fraud_specified'] = (df['any_fraud']==1).astype(int).to_list()
    else:
        print ('please choose one of our three choices above')
    return df2[['fraud_specified']]

fraud_column = fraud_selection(df2, 'any')
with open('fraud_column.pkl', 'wb') as f:
    pickle.dump(fraud_column, f)

any_fraud_mean = df2.any_fraud.mean()
# print ('fraud column mean', fraud_column.values.mean())
# print ('any fraud column mean', any_fraud_mean)
assert fraud_column.values.mean() == any_fraud_mean, 'fraud columns discrepancy...'

account_age = (df2['ca_created_at'] - pd.to_datetime(df2['gr_account_added_date'])).dt.days
account_age_ = account_age.fillna(0).astype(int)
df2.loc[:,'gr_account_added_date'] = account_age_

df2.to_pickle('./df2.pkl')
    
# TIME BINNING AND DATA NORMALIZATION OUTSTEP
# from itertools import product, combinations

# base = ['A','B']
# for n in range(1, len(test_features)+1):   # start at 1 to avoid empty set; end at +1 for full set
#     for q in combinations(range(len(test_features)), n): # range starting at zero covers entire span so +1 not necessary
#         pass
# #         print (pd.concat([ggg[base], ggg[np.take(test_features, list(q))]], axis=1))

# features = ['requested_loan_amount', 'claimed_mni', 'time_spent_on_rates_terms', 'fsl_source',
#             'tmx_raw_true_ip_organization_type', 'tcr_phone_type_description', 'gr_verification_response', 
#             'gr_account_response_code', 'gr_customer_response_code', 'gr_account_added_date']

# features = [f.strip(" ").strip(string.punctuation) for f in features]

# categorical_vars_ = set(features).intersection(categorical_vars)
# all_date_vars_ = set(features).intersection(all_date_vars)
# numeric_vars_ = set(features).intersection(numeric_vars)
# all_vars = categorical_vars_ | all_date_vars_ | numeric_vars_
# print ("\t>> Selected Features to be Transformed:")
# for v in sorted(all_vars):
#     print ('\t\t', v)
# print()
# to_remove = list(set(features).difference(all_vars))
# if len(to_remove) > 0:
#     print ("\t>> Features to be Removed pre-Standardization:\n", to_remove)
#     df2.drop(to_remove, inplace=True, axis=1)
# else:
#     print ("\t>> No Features to be Removed pre-Standardization\n")
    
# df2 = df2[features]

if __name__ == "__main__":

    from itertools import product, combinations

    df2 = pd.read_pickle('./df2.pkl')

    base = ['requested_loan_amount', 'claimed_mni', 'time_spent_on_rates_terms', 'fsl_source',
                'tmx_raw_true_ip_organization_type', 'tcr_phone_type_description', 'gr_verification_response', 
                'gr_account_response_code', 'gr_customer_response_code', 'gr_account_added_date']

    test_features = ['p_tcr_ft_difflib_first_name_match', 'ne_score', 'np_score', 'tmx_true_ip_score', 'tmx_proxy_ip_score', 
                'rr_date_first_seen_ft_days_since', 'tmx_account_email_first_seen_ft_days_since',
               'tmx_ss_ft_true_ip_address_distance', 'rr_popularity', 'ea_score']

    combo_metrics_df = defaultdict()

    for n in range(1, len(test_features)+1):   # start at 1 to avoid empty set; end at +1 for full set
        for q in combinations(range(len(test_features)), n): # range starting at zero covers entire span so +1 not necessary
            testcombo = pd.concat([df2[base], df2[np.take(test_features, list(q))]], axis=1)
            combo_features = list(testcombo.columns)
            print ('testing ', str(n), 'features')
            combo_metrics_df[frozenset(combo_features)] = create_repeat(df2[combo_features], combo_features, ca_created_at, str(int(params.batch_window)))

    with open('combo_metrics_df.pkl', 'wb') as f:
        pickle.dump(combo_metrics_df, f)
    print ('completionn of combo_metrics_df...')


# df2_stand_norm = create_repeat(df2, features, ca_created_at, str(int(params.batch_window)))
# with open('df2_stand_norm.pkl', 'wb') as f:
#     pickle.dump(df2_stand_norm, f)
    
# print ('Standardized and Normalized Dictionary of Time Segments...\n', df2_stand_norm)
# toc = time.perf_counter()
# print (f"\t>> data intake and extraction completed {toc - tic:0.3f} seconds...\n")
