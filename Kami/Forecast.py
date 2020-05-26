'''
This script loads previously trained model to predict new values
'''

import os
import glob
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from .Analyse import Helper
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Forecast:
	def __init__(self, output_dir_path, cache_dir_path, store_list, product_list, start, end, columns = ['store', 'product', 'day_of_week', 'day_of_month', 'year', 'month']):
		print('{0:*^80}'.format('Loading Previously Saved Model...'))
		self.models = [load_model(cache_dir_path + weights_file_name) for weights_file_name in glob.glob('best_model_weights_*.hdf5')]
		with open(cache_dir_path + 'scale_base.txt', 'r') as f:
			self.scale_base = float(f.read())
		print('{0:*^80}'.format('Generating Input Data...'))
		self.generate_input(cache_dir_path, store_list, product_list, start, end, columns)
		print('{0:*^80}'.format('A Sample of Generated Transformed Input Data:'))
		print(self.df_input[:5])
		print('{0:*^80}'.format('Predicting Values Based on Input...'))
		predictions = self.predict(cache_dir_path, self.models, self.df_input)
		self.reformat(cache_dir_path, output_dir_path, predictions, self.df)
		print('{0:*^80}'.format('Ex-ante Predictions Saved to Memory'))

	def generate_input(self, cache_dir_path, store_list, product_list, start, end, columns):
		date = pd.bdate_range(start = start, end = end)
		row_list = []
		for day in date:
			for store in store_list:
				for product in product_list:
					dict = {}
					dict.update({'date': day, 'store': store, 'product': product})
					row_list.append(dict)
		self.df_input = pd.DataFrame(row_list, columns = ['date', 'store', 'product'])
		self.df = self.df_input.copy()
		self.df_input['day_of_week'] = pd.to_datetime(self.df_input['date']).dt.weekday + 1
		self.df_input['day_of_month'] = pd.to_datetime(self.df_input['date']).dt.day
		self.df_input['year'] = pd.to_datetime(self.df_input['date']).dt.year
		self.df_input['month'] = pd.to_datetime(self.df_input['date']).dt.month
		self.df_input.drop('date', axis = 1, inplace = True)
		self.df_input = self.df_input.astype(str)
		self.df_input = self.df_input.to_numpy()
		with open(cache_dir_path + 'les.pickle', 'rb') as f:
			les = pickle.load(f)
		for i in range(self.df_input.shape[1]):
			self.df_input[:, i] = les[i].transform(self.df_input[:, i])
		self.df_input = self.df_input.astype(int)

	def predict(self, cache_dir_path, models, df_input):
		predictions = []
		with open(cache_dir_path + 'ex_ante_predictions.csv', 'w') as f:
			f.write(','.join(Helper.feature_labels) + ',predicted\n')
			for i, record in enumerate(df_input):
				y_pred = np.mean([Aux.guess(model, record, Helper.feature_labels, self.scale_base)[0] for model in models])
				predictions.append(y_pred)
				f.write('{},{},{},{},{},{},{}\n'.format(*[record[i] for i in range(len(Helper.feature_labels))] + [y_pred]))
		return predictions

	def reformat(self, cache_dir_path, output_dir_path, predictions, df):
		df['predicted'] = predictions
		df.to_csv(output_dir_path + 'ex_ante_predictions_untransformed.csv', index = False)

class Aux:
	def guess(model, features, feature_labels, scale_base):
		features = Aux.preprocessing(features, feature_labels)
		result = model.predict(features).flatten()
		return Aux._val_for_pred(result, scale_base)

	def preprocessing(X, feature_labels):
		return Aux.split_features(X, feature_labels)

	def split_features(X, feature_labels):
		X_list = [X[..., [i]] for i in range(len(feature_labels))]
		return X_list

	def _val_for_pred(val, scale_base):
		return np.exp(val * scale_base)
