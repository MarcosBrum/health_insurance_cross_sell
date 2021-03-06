# %% markdown
# Import Modules

# %% codecell
import pandas as pd
from flask import Flask, request, Response
from cross_sell.cross_sell import CrossSell
# o pacote é a pasta. Deste, quer-se o Arquivo, e importa-se a Classe.
import pickle
import os


# %% markdown
# Load model

# %% codecell
model = pickle.load(open('model/xgb_validated_new.pkl', 'rb'))


# %% markdown
# API

# %% codecell
app = Flask(__name__)


@app.route('/cross_sell/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):  # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        else:  # multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instantiate Rossmann class
        pipeline = CrossSell()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, df3)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
