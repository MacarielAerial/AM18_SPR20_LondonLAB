'''
This script prepares raw data to be analysed by neural network later
'''

# Import libraries
import numpy as np
import pandas as pd
import os

class Preprocess:
	'''Main module'''
	def __init__(self, input_f_path, cache_dir_path, split_ratio = 0.95, cols_renames = {'description': 'product', 'day_of_the_week_monday_is_0': 'day_of_week', 'total_average_sell': 'price', 'total_net_sales': 'sales'}):
		'''Initiate settings'''
		pd.options.mode.chained_assignment = None
		self.cols_renames, self.cache_dir_path = cols_renames, cache_dir_path
		self.data_import(input_f_path = input_f_path)
		print('{0:*^80}'.format('Raw Data Imported'))
		self.data_clean(df = self.df_raw, cache_dir_path = cache_dir_path, split_ratio = split_ratio, cols_renames = cols_renames)
		print('{0:*^80}'.format('Raw Data Cleaned'))
		self.shutdown(cache_dir_path = cache_dir_path)
		print('{0:*^80}'.format('Cleaned Data Exported'))

	def __repr__(self):
		return 'Please assign an object to store the instance'

	def data_import(self, input_f_path):
		'''Import sales data'''
		self.df_raw = pd.read_csv(input_f_path)

	def data_clean(self, df, cache_dir_path, split_ratio, cols_renames):
		'''Convert data into the required format'''
		self.train, self.test, self.train_weekly, self.test_weekly, self.df, self.df_weekly = Aux.clean_product_data(df = df, split_ratio = split_ratio, cols_renames = cols_renames)

	def shutdown(self, cache_dir_path):
		'''Export results as the final step'''
		self.train.to_csv(cache_dir_path + 'train.csv', index = False)
		self.test.to_csv(cache_dir_path + 'test.csv', index = False)
		self.train_weekly.to_csv(cache_dir_path + 'train_weekly.csv', index = False)
		self.test_weekly.to_csv(cache_dir_path + 'test_weekly.csv', index = False)
		self.df.to_csv(cache_dir_path + 'df.csv', index = False)
		self.df_weekly.to_csv(cache_dir_path + 'df_weekly.csv', index = False)

class Aux:
	'''Axuliary module to structure the code'''
	def clean_product_data(df, split_ratio, cols_renames, cols_drop = ['date.1', 'week_of_year']):
		'''Specialise in cleaning sales data segmented by store and product'''
		# Clean auxiliary arrays
		df.columns = Helper.clean_col_names(columns = df.columns)
		df.rename(columns = cols_renames, inplace = True)

		# Delete redundant data
		df.drop_duplicates(inplace = True)
		df.drop(columns = cols_drop, inplace = True)

		# Modify raw data
		df = df.loc[df['sales'] > 0.01, :]
		df.loc[:, cols_renames['day_of_the_week_monday_is_0']] = (df[cols_renames['day_of_the_week_monday_is_0']].astype(int) + 1).astype('category')
		df.loc[:, 'date'] = pd.to_datetime(df['date'])
		df.loc[:, 'quantity'] = df['sales']/df['price']

		# Re-order and split modified data
		df = df.set_index(['date', 'store', 'product']).sort_index(ascending = True)
		df_weekly = df.groupby([pd.Grouper(level = 'date', freq = 'W'), pd.Grouper(level = 'store'), pd.Grouper(level = 'product')]).agg({'sales': 'sum', 'price': 'mean', 'quantity': 'sum', 'day_of_week': 'first', 'day_of_month': 'first', 'month': 'first'})
		df.reset_index(inplace = True), df_weekly.reset_index(inplace = True)
		train, test = df.iloc[:round(len(df) * split_ratio), :], df.iloc[round(len(df) * split_ratio):, :]
		train_weekly, test_weekly = df_weekly.iloc[:round(len(df_weekly) * split_ratio), :], df_weekly.iloc[round(len(df_weekly) * split_ratio):, :]

		return train, test, train_weekly, test_weekly, df, df_weekly

class Helper:
	'''Standalone helper function to further reduce clutter'''
	def clean_col_names(columns):
		'''Standardise all column labels'''
		return columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
