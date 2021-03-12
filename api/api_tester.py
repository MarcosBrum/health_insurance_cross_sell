# %% markdown
# Import Modules

# %% codecell
import pandas as pd
import json
import requests


# %% markdown
# Load model

# %% codecell
df = pd.read_csv('/home/marcos/Documentos/comunidade_DS/pa004_health_insurance_cross_sell/data/test.csv')


# %% codecell
data = json.dumps(df.to_dict(orient='records'))


# %% codecell
# API Call
#url = 'http://0.0.0.0:5000/cross_sell/predict'
url = 'https://cross-sell-insurance.herokuapp.com/cross_sell/predict'
header = {'Content-type': 'application/json'}
data = data

r = requests.post(url, data=data, headers=header)
print('Status Code {}'.format(r.status_code))


# %% codecell
d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())
d1 = d1[['id', 'score']]
print(d1)
