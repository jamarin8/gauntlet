from params import params
import pickle
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List

from sklearn.decomposition import PCA

pipInstall = "pip install pandas==1.3.1" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

import pandas as pd
import numpy as np

pipInstall = "pip install pyod" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.models.knn import KNN
# from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
    
import warnings
warnings.filterwarnings('ignore')
    
def create_metrics(df2_stand_norm: Dict, fraud_column: pd.DataFrame, fm_5: pd.DataFrame) -> pd.DataFrame:
    
    rnd_ = 42; contam = .03

    detector_list_1 = [IForest(n_estimators=128), HBOS()]

    detector_list_2 = [LOF(n_neighbors=n_neighbor) for n_neighbor in np.arange(5,21,5)] + \
                        [KNN(method='mean',n_neighbors=n_neighbor) for n_neighbor in [1,3,5,10,15,20]] + \
                        [IForest(n_estimators=128), HBOS()]
    
    classifier_dict = {'IForest (128)': IForest(n_estimators=128), 
                       'IForest (256)': IForest(n_estimators=256),
                       'Histogram Based': HBOS(contamination=contam),
                       'Local Outlier Factor LOF (13)': LOF(n_neighbors=13),
                       'K Nearest Neighbors KNN (13)': KNN(method='mean',n_neighbors=13),
                       'IForest/HBOS Ensembles': LSCP(detector_list_1, contamination=contam, random_state=rnd_),
                       'IForest/HBOS/LOF/KNN Ensembles': LSCP(detector_list_2, contamination=contam, random_state=rnd_)}

    relevant_columns = list(df2_stand_norm[str(int(params.batch_window))].keys())

    print (relevant_columns)

    metrics_df = pd.DataFrame(np.zeros((len(classifier_dict.keys()), len(relevant_columns))), 
                              columns = relevant_columns,
                             index = [classifier_name for classifier_name in classifier_dict.keys()])
    
    window = str(int(params.batch_window))
    
    for d in relevant_columns:
    
        X_train = df2_stand_norm[window][d]

        '''
        Dimension Reduction
        '''
        df = X_train.join(fraud_column).join(fm_5)
        labels = df.fraud_specified
        fm_5_scores = df.fm_5_score
        fm_5_scores = (fm_5_scores - 0.00197736842965041) / 1.22643393399164
        fm_pred = fm_5_scores > 0.05
        df_ = df.drop(['fraud_specified', 'fm_5_score'], axis=1)

        for name, classifier in classifier_dict.items():
            
            if (("IForest" in name) or ("Histogram" in name)) and (not "Ensemble" in name):
                
                print (name, classifier.__class__.__name__, d)
                classifier.fit(X_train)
                classifier_proba = classifier.predict_proba(X_train)
                classifier_pred = classifier_proba[:,1] > 0.7
                augmented_pred = (fm_pred | classifier_pred)
            
            else: 
#             
                print (name, classifier.__class__.__name__, 'design matrix', d)
                pca = PCA(n_components=7)
                design_mtrx = pca.fit_transform(X_train)

                classifier.fit(design_mtrx)
                classifier_proba = classifier.predict_proba(design_mtrx)
                classifier_pred = classifier_proba[:,1] > 0.9
                augmented_pred = (fm_pred | classifier_pred)

            tcf = float("{0:.4f}".format(((fm_pred==0) & (labels==1)).sum() * params.avg_loan_size_fraud)) 
            augmented_tcf = float("{0:.4f}".format(((augmented_pred==0) & (labels==1)).sum() * params.avg_loan_size_fraud))
            tcol = float("{0:.4f}".format(((fm_pred==1) & (labels==0)).sum() * params.avg_conversion_loss_for * params.avg_contribution_profit_percent * params.avg_loan_size_non_fraud))
            augmented_tcol = float("{0:.4f}".format(((augmented_pred==1) & (labels==0)).sum() * params.avg_conversion_loss_for * params.avg_contribution_profit_percent * params.avg_loan_size_non_fraud))
            
            metrics_df.loc[name, d] = (tcf + tcol) - (augmented_tcf + augmented_tcol)
                
    return metrics_df

if __name__ == "__main__":

    with open('df2_stand_norm.pkl', 'rb') as f:
        df2_stand_norm = pickle.load(f)
    with open('fraud_column.pkl', 'rb') as f:
        fraud_column = pickle.load(f)
    with open('fm_5.pkl', 'rb') as f:
        fm_5 = pickle.load(f)

    metrics_df = create_metrics(df2_stand_norm, fraud_column, fm_5)
    print (metrics_df)



    
