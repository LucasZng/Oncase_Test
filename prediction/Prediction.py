import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime
from pyod.models.knn import KNN

class Prediction (object):
    def __init__(self):
        self.home_path = ''
        self.produto_valor_unitario_comercializacao_scaler = pickle.load(open(self.home_path + 'parameter/produto_valor_unitario_comercializacao_scaler.pkl', 'rb') )
        self.year_scaler                                   = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.num_produto_descricao_1_scaler                = pickle.load(open(self.home_path + 'parameter/num_produto_descricao_1_scaler.pkl', 'rb'))
        self.num_produto_descricao_2_scaler                = pickle.load(open(self.home_path + 'parameter/num_produto_descricao_2_scaler.pkl', 'rb'))
        self.num_produto_unidade_comercial_1_scaler        = pickle.load(open(self.home_path + 'parameter/num_produto_unidade_comercial_1_scaler.pkl', 'rb'))
        self.num_produto_unidade_comercial_2_scaler        = pickle.load(open(self.home_path + 'parameter/num_produto_unidade_comercial_2_scaler.pkl', 'rb'))
        
    def data_cleaning (self, df1):
        df1['nota_data_emissao'] =  pd.to_datetime(df1['nota_data_emissao'], format='%Y%m%d %H:%M:%S')
        
        return df1

    def feature_engineering (self, df2):
        # year
        df2['year'] = df2['nota_data_emissao'].dt.year

        # month
        df2['month'] = df2['nota_data_emissao'].dt.month

        # day
        df2['day'] = df2['nota_data_emissao'].dt.day

        # week of year
        df2['week_of_year'] = df2['nota_data_emissao'].dt.isocalendar().week.astype(np.int64)
        
        return df2

    def data_preparation (self, df5):

        # preço
        df5['produto_valor_unitario_comercializacao'] = self.produto_valor_unitario_comercializacao_scaler.transform (df5[['produto_valor_unitario_comercializacao']].values)

        # year
        df5['year'] = self.year_scaler.transform (df5[['year']].values)
        
        # produto_descricao
        df5['num_produto_descricao'] = self.num_produto_descricao_1_scaler.transform(df5['produto_descricao'])
        df5['num_produto_descricao'] = self.num_produto_descricao_2_scaler.transform (df5[['num_produto_descricao']].values)

        # produto_unidade_comercial
        df5['num_produto_unidade_comercial'] = self.num_produto_unidade_comercial_1_scaler.transform(df5['produto_unidade_comercial'])
        df5['num_produto_unidade_comercial'] = self.num_produto_unidade_comercial_2_scaler.transform (df5[['num_produto_unidade_comercial']].values)

        # month
        df5['month_sin'] = df5['month'].apply (lambda x: np.sin (x*2*np.pi/12))
        df5['month_cos'] = df5['month'].apply (lambda x: np.cos (x*2*np.pi/12))

        # day
        df5['day_sin'] = df5['day'].apply (lambda x: np.sin (x*2*np.pi/30))
        df5['day_cos'] = df5['day'].apply (lambda x: np.cos (x*2*np.pi/30))

        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply (lambda x: np.sin (x*2*np.pi/52))
        df5['week_of_year_cos'] = df5['week_of_year'].apply (lambda x: np.cos (x*2*np.pi/52))
        
        cols_selected = ['produto_valor_unitario_comercializacao', 'year', 'num_produto_descricao', 'num_produto_unidade_comercial', 'month_sin', 'month_cos', 'day_sin',
                         'day_cos', 'week_of_year_sin', 'week_of_year_cos']

        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        return original_data.to_json(orient='records', date_format='iso')