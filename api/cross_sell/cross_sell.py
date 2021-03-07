# %% markdown
# Import Modules

# %% codecell
import pandas as pd
import pickle


# %% markdown
# Class

# %% codecell
class CrossSell:
    def __init__(self):
        self.home_path = "/home/marcos/Documentos/comunidade_DS/pa004_health_insurance_cross_sell/"
        self.rs_age = pickle.load(open(self.home_path + "encoders/age_new_scaler.pkl", "rb"))
        self.region_freq = pickle.load(open(self.home_path + "encoders/region_freq_new_scaler.pkl", "rb"))
        self.channel_freq = pickle.load(open(self.home_path + "encoders/channel_freq_new_scaler.pkl", "rb"))
        self.rs_premium = pickle.load(open(self.home_path + "encoders/premium_new_scaler.pkl", "rb"))
        self.mms_vintage = pickle.load(open(self.home_path + "encoders/vintage_new_scaler.pkl", "rb"))
        self.age_encoder = pickle.load(open(self.home_path + "encoders/v_age_new_encoder.pkl", "rb"))
        self.damage_encoder = pickle.load(open(self.home_path + "encoders/v_dam_new_encoder.pkl", "rb"))

    def data_cleaning(self, df1):
        # lowercase columns
        cols_list = df1.columns.tolist()
        cols_lower = [x.lower() for x in cols_list]
        df1.columns = cols_lower

        # data types
        df1['region_code'] = df1['region_code'].astype(int)
        df1['policy_sales_channel'] = df1['policy_sales_channel'].astype(int)
        return df1

    def feature_engineering(self, df2):
        def damage_map(damage):
            if damage == 'Yes':
                return 1
            else:
                return 0

        vehicle_hist = df2['vehicle_damage'].map(damage_map) + 1 - df2['previously_insured']

        df2.insert(loc=len(df2.columns)-1, column='vehicle_hist', value=vehicle_hist)
        return df2

    def data_preparation(self, df9):
        df9 = df9.copy()

        # apply encoders and scalers
        #gender: "One-hot" encoder
        df9 = pd.get_dummies(df9, prefix='gender', columns=['gender'])

        # age - RobustScaler
        df9['age'] = self.rs_age.transform(df9[['age']].values)

        # region_code - frequency encoding
        df9['region_code'] = df9['region_code'].map(self.region_freq)

        # policy_sales_channel - frequency encoding
        df9['policy_sales_channel'] = df9['policy_sales_channel'].map(self.channel_freq)

        # annual_premium - RobustScaler
        df9['annual_premium'] = self.rs_premium.transform(df9[['annual_premium']].values)

        # vintage - MinMaxScaler
        df9['vintage'] = self.mms_vintage.transform(df9[['vintage']].values)

        # vehicle_age - LabelEncoder
        df9['vehicle_age'] = self.age_encoder.transform(df9['vehicle_age'])

        # vehicle_damage - LabelEncoder
        df9['vehicle_damage'] = self.damage_encoder.transform(df9['vehicle_damage'])

        # select columns
        cols_selected_full = ['id', 'age', 'policy_sales_channel', 'previously_insured', 'annual_premium', 'vintage',
                              'vehicle_hist', 'gender_Female', 'gender_Male']
        df9 = df9[cols_selected_full].copy()
        return df9

    def get_prediction(trained_model, data_test):
        # drop id
        data_testing = data_test.drop(['id'], axis=1).copy()
        # predict_proba:
        yhat_proba = trained_model.predict_proba(data_testing)

        # transform yhat_proba to 1D-array
        yhat_proba_1d = yhat_proba[:, 1].tolist()

        # include in dataframe
        testing_data = data_test.copy()
        testing_data['score'] = yhat_proba_1d
        # sort
        testing_data = testing_data.sort_values('score', ascending=False)
        # reset index
        testing_data.reset_index(drop=True, inplace=True)
        return testing_data.to_json(orient='records')
