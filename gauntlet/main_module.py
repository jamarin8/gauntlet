import pickle
import subprocess, os
from collections import defaultdict, Counter

pipInstall = "pip install pandas==1.3.1" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

pipInstall = "pip install openpyxl" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

pipInstall = "pip install xlsxwriter"
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

import pandas as pd
import numpy as np

from params import params
from gauntlet import create_metrics

# if os.path.exists('./gauntlet.py')

from dataintake import fraud_column, fm_5
from dataintake import df2_stand_norm, any_fraud_mean

print ('\t>> time segments:')
print ('\t\t', [dat.strftime('%Y-%m-%d') for dat in df2_stand_norm[str(params.batch_window)].keys()], '\n')
print ('\t>> fraud rate in full sample: ', np.round(any_fraud_mean, 5))

if params.demo_mode == False:
    
    with open('normalized_repeat_dict.pkl', 'rb') as f:
        normalized_repeat_dict = pickle.load(f)
        
    with open('fraud_column.pkl', 'rb') as f:
        fraud_column = pickle.load(f)
        
    with open('fm_5.pkl', 'rb') as f:
        fm_5 = pickle.load(f)
        
else:
    
    metrics_df = create_metrics(df2_stand_norm, fraud_column, fm_5)
    print (metrics_df)
    
    writer = pd.ExcelWriter('metrics_df.xlsx') 
    metrics_df.to_excel(writer, sheet_name='cost_analysis', index=True, na_rep='NaN')

    # Auto-adjust columns' width
    for column in metrics_df:
        column_width = max(metrics_df[column].astype(str).map(len).max(), len(column)) + 1
        col_idx = metrics_df.columns.get_loc(column)
        writer.sheets['cost_analysis'].set_column(col_idx, col_idx, column_width)

    # Auto-adjust index width
    from gauntlet import classifier_dict

    writer.sheets['cost_analysis'].set_column(0, 0,  max([len(classifier_name) for classifier_name in classifier_dict.keys()]) + 1)

    # Auto-adjust last column width
    last = metrics_df.columns.get_loc(metrics_df.columns[-1])
    writer.sheets['cost_analysis'].set_column(last+1, last+1, max(metrics_df[metrics_df.columns[-1]].astype(str).map(len).max(), len(column)) + 1)
    writer.save()

