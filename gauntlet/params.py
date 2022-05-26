from dataclasses import dataclass, fields, make_dataclass

print ("\nPARAMETER INTAKE MODULE...")

import subprocess
pipInstall = "pip install pandas" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

pipInstall = "pip install openpyxl" 
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

pipInstall = "/opt/conda/bin/python -m pip install --upgrade pip"
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

import pandas as pd, os

path = os.path.abspath('./')

import warnings
warnings.filterwarnings('ignore')

# working_directory = os.cwd()

def param_prep(key_intake: str) -> str:
    if len(key_intake.split()) == 1:
        if '-' in key_intake:
            key_intake = key_intake.replace('-', '_')
        return key_intake.lower()
    else:
        compact_str = "_".join(list(map(lambda x: x.lower(), key_intake.split()[:4])))
        compact_str = compact_str.replace('(','').replace(')','').replace('-', '_')
        return compact_str

intake_params = pd.read_excel(path + '/settings.xlsx', sheet_name = 'Inputs', header=None, index_col=0, engine='openpyxl').to_dict()[1]

slot_dict = {k:v for k,v in zip(list(map(param_prep, intake_params)), intake_params.values())}

Params = make_dataclass('Params', list(slot_dict.keys()))
params = Params(*slot_dict.values())

if __name__ == "__main__":
    print (params)






