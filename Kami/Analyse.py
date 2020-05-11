'''
This script applies Entity Embedding neural network pioneered by Neokami during the Rossman Challenge in Kaggle to POS data of a hospitality firm
'''

# Import libraries
import numpy as np
import pandas as pd
import pickle
import os
import csv
import sys
from .EntityEmbedding import EntityEmbedding
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
sys.setrecursionlimit(10000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Analyse:
	'''Main module'''
	def __init__(self, output_dir_path, cache_dir_path, sales_as_label = True, weekly_agg = False, n_1 = 1024, n_2 = 512, n_3 = 256, n_4 = 128, n_5 = 64, dropout = 0.4, output_activation = 'relu', err_func = 'mean_absolute_error', optimizer = 'adam', epochs = 25, patience = 5, batch_size = 256, n_sample = 600000, n_ensemble = 5, val_split_ratio = 0.95, save_embeddings = True, saved_embeddings_fname = 'embeddings.pickle'):
		'''Initiate local variables'''
		self.r_train, self.r_val = 0, 0
		target_label = 'sales' if sales_as_label else 'quantity'
		print('{0:*^80}'.format('Sales Forecast with Entity Embedding Model Initiated'))
		self.extract_csv(cache_dir_path, weekly_agg = weekly_agg)
		self.prep_features(cache_dir_path, target_label = target_label)
		models = self.train_model(cache_dir_path, output_dir_path, n_1, n_2, n_3, n_4, n_5, dropout, output_activation, err_func, optimizer, epochs, patience, batch_size, n_sample, n_ensemble, val_split_ratio, save_embeddings, saved_embeddings_fname)
		self.test_model(models, cache_dir_path, output_dir_path)
		print('{0:*^80}'.format('Sales Forecast with Entity Embedding Model Completed'))

	def __repr__(self):
		return 'Please assign an object to store the instance'

	def extract_csv(self, cache_dir_path, weekly_agg):
		'''Convert cached csv files into dictionary-like objects'''
		train_path = (cache_dir_path + 'train.csv') if not weekly_agg else (cache_dir_path + 'train_weekly.csv')
		test_path = (cache_dir_path + 'test.csv') if not weekly_agg else (cache_dir_path + 'test_weekly.csv')
		with open(train_path) as csv_train, open(test_path) as csv_test:
			train, test = csv.reader(csv_train, delimiter = ','), csv.reader(csv_test, delimiter = ',')
			with open(cache_dir_path + 'train.pickle', 'wb') as f_train, open(cache_dir_path + 'test.pickle', 'wb') as f_test:
				train, test = Helper.csv2dict(train), Helper.csv2dict(test)
				train = train[::-1]
				pickle.dump(train, f_train, -1), pickle.dump(test, f_test, -1)

	def prep_features(self, cache_dir_path, target_label):
		'''Engineer features to ready for neural network'''
		with open(cache_dir_path + 'train.pickle', 'rb') as f_train, open(cache_dir_path + 'test.pickle', 'rb') as f_test:
			train, test = pickle.load(f_train), pickle.load(f_test)
			n_train_val = len(train)

		train_x, train_y, test_x, test_y = [], [], [], []
		train_x, train_y = Aux.select_and_split(data = train, target = train_y, features = train_x, target_label = target_label)
		test_x, test_y = Aux.select_and_split(data = test, target = test_y, features = test_x, target_label = target_label)
		print('{0:*^80}'.format('Number of Train & Validation and Test Observations:'))
		print('{0:*^80}'.format(str(len(train_y)) + ' and ' + str(len(test_y))))
		print('{0:*^80}'.format('Range of Train Target:'))
		print('{0:*^80}'.format(str(min(train_y)) + ' to ' + str(max(train_y))))
		train_x, test_x = Aux.encode_labels(features = train_x, cache_dir_path = cache_dir_path), Aux.encode_labels(features = test_x, cache_dir_path = cache_dir_path)

		with open(cache_dir_path + 'train_prepped.pickle', 'wb') as f_train, open(cache_dir_path + 'test_prepped.pickle', 'wb') as f_test:
			pickle.dump((train_x, train_y), f_train, -1), pickle.dump(test_x, f_test, -1)
		pd.DataFrame(test_x).to_csv(cache_dir_path + 'test_features_encoded.csv', header = Helper.feature_labels, index = False)

	def train_model(self, cache_dir_path, output_dir_path, n_1, n_2, n_3, n_4, n_5, dropout, output_activation, err_func, optimizer, epochs, patience, batch_size, n_sample, n_ensemble, val_split_ratio, save_embeddings, saved_embeddings_fname):
		'''Train an entity embedding LSTM neural network to predict target variable'''
		with open(cache_dir_path + 'train_prepped.pickle', 'rb') as f:
			(X, y) = pickle.load(f)
		X_train, X_val, y_train, y_val = Aux.train_val_split(X, y, val_split_ratio, n_sample)
		print('{0:*^80}'.format('Number of Train Observations:'))
		print('{0:*^80}'.format(str(y_train.shape[0])))

		print('{0:*^80}'.format('Fitting Neural Network with Entity Embedding LSTM...'))
		models = []
		[models.append(EntityEmbedding(X_train, y_train, X_val, y_val,
						 cache_dir_path, output_dir_path,
						 Helper.feature_labels,
						 n_1, n_2, n_3, n_4, n_5,
						 dropout, output_activation,
						 err_func, optimizer, epochs,
						 patience, batch_size)) for i in range(n_ensemble)]
		if save_embeddings:
			Helper.save_embeddings(models, cache_dir_path)

		print('{0:*^80}'.format('Evaluating the Ensemble Model'))
		print('{0:*^80}'.format('Training Error:'))
		self.r_train = Aux.evaluate_models(models, X_train, y_train)
		print('{0:*^80}'.format(str(self.r_train)))
		print('{0:*^80}'.format('Validation Error:'))
		self.r_val = Aux.evaluate_models(models, X_val, y_val)
		print('{0:*^80}'.format(str(self.r_val)))
		return models

	def test_model(self, models, cache_dir_path, output_dir_path):
		'''Evaluate model performance based on test data'''
		print('{0:*^80}'.format('Exporting Predictions to Memory...'))
		with open(cache_dir_path + 'test_prepped.pickle', 'rb') as f:
			X_test = pickle.load(f)
		with open(output_dir_path + 'test_predicted.csv', 'w') as f:
			f.write(','.join(Helper.feature_labels) + ',predicted\n')
			for i, record in enumerate(X_test):
				y_pred = np.mean([model.guess(record, Helper.feature_labels) for model in models])
				f.write('{},{},{},{},{},{},{}\n'.format(*[record[i] for i in range(len(Helper.feature_labels))] + [y_pred]))

class Aux:
	'''Auxiliary module to reduce code clutters'''
	def select_and_split(data, target, features, target_label):
		'''Select a subset of features and return separate arrays for target and features'''
		for i in range(len(data)):
			fl = Helper.select_features(record = data[i])
			features.append(fl)
			target.append(float(data[i][target_label]))
		features, target = np.array(features), np.array(target)
		return features, target

	def encode_labels(features, cache_dir_path):
		'''Encode categorical features with integers'''
		les = []
		for i in range(features.shape[1]):
			le = LabelEncoder()
			le.fit(features[:, i])
			les.append(le)
			features[:, i] = le.transform(features[:, i])
		features = features.astype(int)

		with open(cache_dir_path + 'les.pickle', 'wb') as f:
			pickle.dump(les, f, -1)
		return features

	def train_val_split(X, y, val_split_ratio, n_sample):
		'''Split data into train and validation datasets and perform random sampling'''
		n_train_val_prepped = len(X)
		n_train = int(n_train_val_prepped * val_split_ratio)
		X_train, X_val, y_train, y_val = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
		X_train, y_train = Helper.sample(X_train, y_train, n_sample)
		return X_train, X_val, y_train, y_val

	def evaluate_models(models, X, y):
		assert(min(y) > 0)
		guessed_sales = np.array([model.guess(X, Helper.feature_labels) for model in models])
		mean_sales = guessed_sales.mean(axis = 0)
		relative_err = np.absolute((y - mean_sales) / y)
		result = np.sum(relative_err) / len(y)
		return result

class Helper:
	'''Independent auxiliary functions'''
	feature_labels = ['store', 'product', 'day_of_week', 'day_of_month', 'year', 'month']

	def csv2dict(csv):
		dict, keys = [], []
		for row_idx, row in enumerate(csv):
			if row_idx == 0:
				keys = row
				continue
			dict.append({key: value for key, value in zip(keys, row)})
		return dict

	def sample(X, y, n):
		'''Randomly sample from given distributions'''
		n_row = X.shape[0]
		indices = np.random.randint(n_row, size = n)
		return X[indices, :], y[indices]

	def save_embeddings(models, cache_dir_path):
		'''Save categorical data embeddings to memory'''
		model = models[0].model
		store_embedding = model.get_layer('store_embedding').get_weights()[0]
		product_embedding = model.get_layer('product_embedding').get_weights()[0]
		dow_embedding = model.get_layer('day_of_week_embedding').get_weights()[0]
		dom_embedding = model.get_layer('day_of_month_embedding').get_weights()[0]
		year_embedding = model.get_layer('year_embedding').get_weights()[0]
		month_embedding = model.get_layer('month_embedding').get_weights()[0]
		with open(cache_dir_path + 'embeddings.pickle', 'wb') as f:
			pickle.dump([store_embedding, product_embedding, dow_embedding, dom_embedding, year_embedding, month_embedding], f, -1)

	def select_features(record):
		'''Select features before separating features from target'''
		dt = datetime.strptime(record['date'], '%Y-%m-%d')
		store = str(record['store'])
		product = str(record['product'])
		day_of_week = int(record['day_of_week'])
		day_of_month =  int(record['day_of_month'])
		year = dt.year
		month = int(record['month'])
		return [store, product, day_of_week, day_of_month, year, month]

