from categorical_features import categorical_vars, all_date_vars, numeric_vars
import subprocess, time, string, pickle
from collections import defaultdict, Counter

pipInstall = "pip install pandas==1.3.1" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np

from time_bin_normalize import create_repeat
from gauntlet import create_metrics
from params import params

# TIME BINNING AND DATA NORMALIZATION OUTSTEP
from itertools import product, combinations

base = ['requested_loan_amount', 'claimed_mni', 'time_spent_on_rates_terms', 'fsl_source',
            'tmx_raw_true_ip_organization_type', 'tcr_phone_type_description', 'gr_verification_response', 
            'gr_account_response_code', 'gr_customer_response_code', 'gr_account_added_date']
'''
the following represents a comprehensive list of available correlated features that can be added

# test_features = ['p_tcr_ft_difflib_first_name_match', 'ne_score', 'np_score', 'tmx_true_ip_score', 'tmx_proxy_ip_score', 
#             'rr_date_first_seen_ft_days_since', 'tmx_account_email_first_seen_ft_days_since',
#            'tmx_ss_ft_true_ip_address_distance', 'rr_popularity', 'ea_score']

'''
test_features = ['ne_score', 'np_score', 'ea_score']

df2 = pd.read_pickle('./df2.pkl')

ca_created_at = pd.read_pickle('./ca_created_at.pkl')

with open('fraud_column.pkl', 'rb') as f:
    fraud_column = pickle.load(f)
    
with open('fm_5.pkl', 'rb') as f:
    fm_5 = pickle.load(f)

combo_metrics_df = defaultdict()
 
for n in range(1, len(test_features)+1):   # start at 1 to avoid empty set; end at +1 for full set
    for q in combinations(range(len(test_features)), n): # range starting at zero covers entire span so +1 not necessary
        testcombo = pd.concat([df2[base], df2[np.take(test_features, list(q))]], axis=1)
        combo_features = list(testcombo.columns)
        print (list(combo_features))        
        print ('creating new chronological dictionary...')
        df2_stand_norm_combo = create_repeat(df2[combo_features], combo_features, ca_created_at, str(int(params.batch_window)))
        print ('getting metrics from models...')
        combo_metrics_df[frozenset(combo_features)] = create_metrics(df2_stand_norm_combo, fraud_column, fm_5)

with open('combo_metrics_df.pkl', 'wb') as f:
    pickle.dump(combo_metrics_df, f)
print ('completionn of combo_metrics_df...')