'''
This script is an attempt in applying Neokami's categorical entity embedding to sales forecast
'''

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import csv
import sys
import config
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Activation, Reshape, Concatenate, Embedding, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
sys.setrecursionlimit(10000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Main:
	'''Main execution module'''
	def __init__(self,):
		'''Initiate local variables'''
		self.input_f_path = config.temp_path + 'sales_data_by_product_stacked_train.csv'
		self.input_f_path_test = config.temp_path + 'sales_data_by_product_stacked_test.csv'

	#def label_encode(self):
	#	df = pd.read_csv(self.input_f_path, index_col = 'date', dtype = {'store': 'category', 'product': 'category', 'sales': 'float64', 'price': 'float64', 'day_of_month': 'int64', 'day_of_the_week': 'int64', 'week_of_year': 'int64', 'month': 'int64'})
	#	self.category_features = df.select_dtypes('category').columns
	#	self.number_features = df.select_dtypes('number').columns
	#	print(self.category_features)
	#	print(self.number_features)
	#	for col in self.category_features:
	#		encoder = LabelEncoder()
	#		df[col] = encoder.fit_transform(df[col])
	#	df.to_csv(config.temp_path + 'sales_data_by_product_stacked_encoded.csv', index = 'date')
	
	def extract_csv(self):
		with open(self.input_f_path) as csvfile:
			data = csv.reader(csvfile, delimiter = ',')
			with open(config.output_path + 'train_data.pickle', 'wb') as f:
				data = Aux.csv2dicts(data)
				data = data[::-1]
				pickle.dump(data, f, -1)
				print(data[:3])

		with open(self.input_f_path_test) as csvfile:
			data = csv.reader(csvfile, delimiter = ',')
			with open(config.output_path + 'test_data.pickle', 'wb') as f:
				data = Aux.csv2dicts(data)
				pickle.dump(data, f, -1)
				print(data[0])

	def prep_features(self):
		with open(config.output_path + 'train_data.pickle', 'rb') as f:
			data = pickle.load(f)
			num_records = len(data)
		with open(config.output_path + 'test_data.pickle', 'rb') as f:
			test_data = pickle.load(f)
		data_x, data_y = [], []

		for record in data:
			fl = Aux.feature_list(record)
			data_x.append(fl)
			data_y.append(float(record['sales']))
		print('Number of train data points: ', len(data_y))

		test_data_x, test_data_y = [], []
		for record in test_data:
			fl = Aux.feature_list(record)
			test_data_x.append(fl)
			test_data_y.append(float(record['sales']))
		print('Number of test data points: ', len(test_data_y))
		print(min(data_y), max(data_y))

		data_x = np.array(data_x)
		print(data_x[:3])
		les = []
		for i in range(data_x.shape[1]):
			le = LabelEncoder()
			le.fit(data_x[:, i])
			les.append(le)
			data_x[:, i] = le.transform(data_x[:, i])

		test_data_x = np.array(test_data_x)
		for i in range(test_data_x.shape[1]):
			le = LabelEncoder()
			le.fit(test_data_x[:, i])
			test_data_x[:, i] = le.transform(test_data_x[:, i])

		print(data_x[:3])
		with open(config.output_path + 'les.pickle', 'wb') as f:
			pickle.dump(les, f, -1)
		data_x = data_x.astype(int)
		data_y = np.array(data_y)
		test_data_x = test_data_x.astype(int)

		with open(config.output_path + 'feature_data.pickle', 'wb') as f:
			pickle.dump((data_x, data_y), f, -1)
			print(data_x[0], data_y[0])
		with open(config.output_path + 'feature_test_data.pickle', 'wb') as f:
			pickle.dump(test_data_x, f, -1)
		a_df = pd.read_csv(config.temp_path + 'sales_data_by_product_stacked_test.csv')
		pd.DataFrame({'date':a_df['date'], 'product': a_df['product'], 'actual':test_data_y}).to_csv(config.output_path + 'actual_test_data.csv', index = False)

	def train_test_model(self, train_ratio = 0.9):
		shuffle_data = False
		one_hot_as_input = False
		embeddings_as_input = False
		save_embeddings = True
		saved_embeddings_fname = 'embeddings.pickle'

		f = open(config.output_path + 'feature_data.pickle', 'rb')
		(X, y) = pickle.load(f)
		num_records = len(X)
		train_size = int(train_ratio * num_records)
		X_train = X[:train_size]
		X_val = X[train_size:]
		y_train = y[:train_size]
		y_val = y[train_size:]
		X_train, y_train = Aux.sample(X_train, y_train, 500000)
		print('Number of samples used for training: ' + str(y_train.shape[0]))
		models = []
		print('Fitting NN_with_EntityEmbedding...')
		for i in range(5):
			models.append(NN_with_EntityEmbedding(X_train, y_train, X_val, y_val))

		if save_embeddings:
			model = models[0].model
			store_embedding = model.get_layer('store_embedding').get_weights()[0]
			product_embedding = model.get_layer('product_embedding').get_weights()[0]
			dow_embedding = model.get_layer('day_of_week_embedding').get_weights()[0]
			dom_embedding = model.get_layer('day_of_month_embedding').get_weights()[0]
			year_embedding = model.get_layer('year_embedding').get_weights()[0]
			month_embedding = model.get_layer('month_embedding').get_weights()[0]
		with open(config.output_path + 'embeddings.pickle', 'wb') as f:
			pickle.dump([store_embedding, product_embedding, dow_embedding, dom_embedding, year_embedding, month_embedding], f, -1)

		print('Evaluate combined models...')
		print('Training error...')
		r_train = Aux.evaluate_models(models, X_train, y_train)
		print(r_train)

		print('Validation error...')
		r_val = Aux.evaluate_models(models, X_val, y_val)
		print(r_val)

		with open(config.output_path + 'feature_test_data.pickle', 'rb') as f:
			test_x = pickle.load(f)

		with open(config.output_path + 'predicted_test_data.csv', 'w') as f:
			f.write('store,product,day,month,year,predicted\n')
			for i, record in enumerate(test_x):
				guessed_sales = np.mean([model.guess(record) for model in models])
				f.write('{},{},{},{},{},{}\n'.format(record[0], record[1], record[3], record[5], record[4], guessed_sales))			

	def exec(self):
		self.extract_csv()
		self.prep_features()
		self.train_test_model(train_ratio = config.train_ratio)
		print('{0:*^80}'.format('Sales Forecast with Entity Embedding Model Completed'))

class Aux:
	'''Auxiliary module'''
	def csv2dicts(csvfile):
		data, keys = [], []
		for row_idx, row in enumerate(csvfile):
			if row_idx == 0:
				keys = row
				print(row)
				continue
			data.append({key: value for key, value in zip(keys, row)})
		return data

	def feature_list(record):
		dt = datetime.strptime(record['date'], '%Y-%m-%d')
		store = str(record['store'])
		product = str(record['product'])
		year = dt.year
		month = int(record['month'])
		day_of_week = int(record['day_of_the_week'])
		day_of_month =  int(record['day_of_month'])
		price = float(record['price'])
		return [store, product, day_of_week, day_of_month, year, month]

	def sample(X, y, n):
		'''random samples'''
		num_row = X.shape[0]
		indices = np.random.randint(num_row, size = n)
		return X[indices, :], y[indices]

	def split_features(X):
		X_list = []

		store = X[..., [0]]
		X_list.append(store)

		product = X[..., [1]]
		X_list.append(product)

		day_of_week = X[..., [2]]
		X_list.append(day_of_week)

		day_of_month = X[..., [3]]
		X_list.append(day_of_month)

		year = X[..., [4]]
		X_list.append(year)

		month = X[..., [5]]
		X_list.append(month)

		return X_list				

	def evaluate_models(models, X, y):
		assert(min(y) > 0)
		guessed_sales = np.array([model.guess(X) for model in models])
		mean_sales = guessed_sales.mean(axis = 0)
		relative_err = np.absolute((y - mean_sales) / y)
		result = np.sum(relative_err) / len(y)
		return result

class Model(object):
	def evaluate(self, X_val, y_val):
		assert(min(y_val) > 0)
		guessed_sales = self.guess(X_val)
		relative_err = np.absolute((y_val - guessed_sales)/y_val)
		result = np.sum(relative_err)/len(y_val)
		return result

class NN_with_EntityEmbedding(Model):
	def __init__(self, X_train, y_train, X_val, y_val):
		super().__init__()
		self.epochs = 15
		self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))
		self.__build_keras_model()
		self.fit(X_train, y_train, X_val, y_val)

	def preprocessing(self, X):
		X_list = Aux.split_features(X)
		return X_list

	def __build_keras_model(self):
		input_store = Input(shape = (1,))
		output_store = Embedding(6, 5, name = 'store_embedding')(input_store)
		output_store = Reshape(target_shape = (5,))(output_store)
		
		input_product = Input(shape = (1,))
		output_product = Embedding(710, 200, name = 'product_embedding')(input_product)
		output_product = Reshape(target_shape = (200,))(output_product)

		input_dom = Input(shape = (1,))
		output_dom = Embedding(31, 10, name = 'day_of_month_embedding')(input_dom)
		output_dom = Reshape(target_shape = (10,))(output_dom)

		input_dow = Input(shape = (1,))
		output_dow = Embedding(7, 6, name = 'day_of_week_embedding')(input_dow)
		output_dow = Reshape(target_shape = (6,))(output_dow)

		input_month = Input(shape = (1,))
		output_month = Embedding(12, 6, name = 'month_embedding')(input_month)
		output_month = Reshape(target_shape = (6,))(output_month)

		input_year = Input(shape = (1,))
		output_year = Embedding(5, 4, name = 'year_embedding')(input_year)
		output_year = Reshape(target_shape = (4,))(output_year)

		input_model = [input_store, input_product, input_dow, input_dom, input_year, input_month]	
		output_embeddings = [output_store, output_product, output_dow, output_dom, output_year, output_month]

		output_model = Concatenate()(output_embeddings)
		output_model = Dropout(0.02)(output_model)
		output_model = Reshape(target_shape = (1,231))(output_model)
		output_model = LSTM(512, return_sequences = True, dropout = 0.4)(output_model)
		output_model = LSTM(256, return_sequences = True, dropout = 0.4)(output_model)
		output_model = LSTM(256, return_sequences = True, dropout = 0.4)(output_model)
		output_model = LSTM(128, return_sequences = True, dropout = 0.4)(output_model)
		output_model = LSTM(64, return_sequences = False, dropout = 0.4)(output_model)
		output_model = Dense(1, activation = 'relu')(output_model)

		self.model = KerasModel(inputs = input_model, outputs = output_model)
		self.model.compile(loss = 'mean_squared_error', optimizer = 'adam')

		plot_model(self.model, to_file = config.output_path + 'entity_embedding_model.png', show_shapes = True, dpi = 300)

	def _val_for_fit(self, val):
		val = np.log(val)/self.max_log_y
		return val

	def _val_for_pred(self, val):
		val = np.exp(val * self.max_log_y)
		return val

	def fit(self, X_train, y_train, X_val, y_val, patience = 3):
		callbacks = [EarlyStopping(monitor = 'val_loss',
						 patience = patience),
						 ModelCheckpoint(filepath = config.output_path + 'best_model_weights.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)
						 ]
		self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
				 validation_data = (self.preprocessing(X_val), self._val_for_fit(y_val)),
				 epochs = self.epochs, batch_size = 128)
		print('Result on validation data: ', self.evaluate(X_val, y_val))

	def guess(self, features):
		features = self.preprocessing(features)
		result = self.model.predict(features).flatten()
		return self._val_for_pred(result)


if __name__ == '__main__':
	obj = Main()
	obj.exec()
