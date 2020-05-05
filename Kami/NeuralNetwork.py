'''
This script contains model specifications
'''

# Import libraries
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Activation, Reshape, Concatenate, Embedding, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

class EntityEmbedding:
	'''Main model instance'''
	def __init__(self, X_train, y_train, X_val, y_val, cache_dir_path, output_dir_path, feature_labels, epochs = 15, patience = 4):
		self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))
		self.__build_keras_model(output_dir_path)
		self.fit(X_train, y_train, X_val, y_val, cache_dir_path, feature_labels, patience = patience, epochs = epochs)

	def preprocessing(self, X, feature_labels):
		return Aux.split_features(X, feature_labels)

	def _val_for_fit(self, val):
		return np.log(val) / self.max_log_y

	def _val_for_pred(self, val):
		return np.exp(val * self.max_log_y)

	def fit(self, X_train, y_train, X_val, y_val, cache_dir_path, feature_labels, patience = 5, epochs = 15):
		callbacks = [EarlyStopping(monitor = 'val_loss', patience = patience), ModelCheckpoint(filepath = cache_dir_path + 'best_model_weights.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)]
		self.model.fit(self.preprocessing(X_train, feature_labels), self._val_for_fit(y_train),
				validation_data = (self.preprocessing(X_val, feature_labels), self._val_for_fit(y_val)),
				epochs = epochs, batch_size = 128)
		print('{0:*^80}'.format('Result on Validation Data:'))
		print('{0:*^80}'.format(self.evaluate(X_val, y_val)))

	def guess(self, features):
		features = self.preprocessing(features)
		result = self.model.predict(features).flatten()
		return self._val_for_pred(result)

	def evaluate(self, X_val, y_val):
		assert(min(y_val) > 0)
		guess_sales = self.guess(X_val)
		relative_err = np.absolute((y_val - guessed_sales)/y_val)
		result = np.sum(relative_err)/len(y_val)
		return result

	def __build_keras_model(self, output_dir_path):
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

		plot_model(self.model, to_file = output_dir_path + 'entity_embedding_model.png', show_shapes = True, dpi = 300)

class Aux:
	def split_features(X, feature_labels):
		X_list = [X[..., [i]] for i in range(len(feature_labels))]
		return X_list
