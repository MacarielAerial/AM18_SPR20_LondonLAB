'''
This script prepares raw data to be analysed by neural network later
'''

# Import libraries
import numpy as np
import pandas as pd
import os

class Preprocess;
	'''Main module'''
	def __init__(self, input_f_path, cache_path, split_ratio = 0.95):
		'''Initiate settings'''
		self.df_raw, self.train, self.test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
		self.data_import(input_f_path = input_f_path)
		print('{0:*^80}'.format('Raw Data Imported'))
		self.data_clean(df = self.df_raw, cache_path = cache_path, split_ratio = split_ratio)
		print('{0:*^80}'.format('Raw Data Cleaned'))
		self.shutdown()
		print('{0:*^80}'.format('Cleaned Data Exported'))

	def data_import(self, input_f_path):
		'''Import sales data'''
		self.df_raw = pd.read_csv(input_f_path)

	def data_clean(self, df, cache_path, split_ratio):
		'''Convert data into the required format'''
		self.train, self.test = Aux.clean_product_data(df = df
								cache_path = cache_path,
								split_ratio = split_ratio
								)

	def shutdown(self, cache_path):
		'''Export results as the final step'''
		self.train.to_csv(cache_path, index = False)
		self.test.to_csv(cache_path, index = False)

class Aux:
	'''Axuliary module to structure the code'''
	def clean_product_data(df, cache_path, split_ratio, cols_drop = ['date.1', 'week_of_year'], cols_renames = {'day_of_the_week_monday_is_0': 'day_of_week', 'description': 'product', 'total_average_sell': 'price', 'total_net_sales': 'sales'}):
		'''Specialise in cleaning sales data segmented by store and product'''
		# Clean auxiliary arrays
		df.columns = Helper.clean_col_names(columns = df.columns)
		df.rename(columns = cols_renames, inplace = True)

		# Delete redundant data
		df.drop_duplicates(inplace = True)
		df.drop(columns = cols_drop, inplace = True)

		# Modify raw data
		df = df.loc[df['sales'] > 0, :]
		df[cols_renames['day_of_the_week_monday_is_0']] = (df[cols_renames['day_of_the_week_monday_is_0']].astype(int) + 1).astype('category')
		df['date'] = pd.to_datetime(df['date'])

		# Re-order and split modified data
		df = df.set_index(['date', 'store', 'product']).sort_index().reset_index()
		train, test = df.iloc[:round(len(df) * split_ratio), :], df.iloc[round(len(df) * split_ratio), :]

		return train, test

class Helper:
	'''Standalone helper function to further reduce clutter'''
	def clean_col_names(columns):
		'''Standardise all column labels'''
		return columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
