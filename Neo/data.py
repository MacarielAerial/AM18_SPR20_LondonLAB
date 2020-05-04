'''
Downloads weather data online and import sales data from memory before structuring these data
'''

# Import libraries
import numpy as np
import pandas as pd
import requests
import dropbox
import os
import config

# Modules
class Data:
	'''Manipulates data from web sources and local memory'''
	def __init__(self, input_path, temp_path, output_path, weather_url, dbx_token, dropbox_path, input_f_names, input_w_name, web_source = False, dev_mode = False, train_ratio = 0.95):
		self.input_path, self.temp_path, self.output_path, self.weather_url, self.dbx_token, self.dropbox_path, self.input_f_names, self.input_w_name, self.web_source, self.dev_mode, self.train_ratio = input_path, temp_path, output_path, weather_url, dbx_token, dropbox_path, input_f_names, input_w_name, web_source, dev_mode, train_ratio
		self.df_sales = pd.DataFrame()
		self.df_weather = pd.DataFrame()
		self.dbx = dropbox.Dropbox(self.dbx_token)
		print('{0:*^80}'.format('Data Cleaning Process Initiated'))

	def authenticate(self):
		'''Authenticate access to data'''
		print('{0:*^80}'.format('This dropbox access token belongs to'))
		print('{0:*^80}'.format(self.dbx.users_get_current_account().name.display_name)) # Authenticate dropbox access

	def data_import(self):
		'''Download data from web sources or import from local memory'''
		self.df_weather = pd.read_csv(self.weather_url) if self.web_source else pd.read_csv(self.input_path + self.input_w_name)
		self.df_sales = pd.read_csv(self.dropbox_path + self.input_f_names[0])
		self.df_sales_product = pd.read_csv(self.dropbox_path + self.input_f_names[1]) # Import two sources of data

	def data_clean(self):
		'''Restructure data for later analysis'''
		self.df_sales = Aux.clean_client_data(self.df_sales)
		self.df_sales_product_train, self.df_sales_product_test = Aux.clean_client_data_product(self.df_sales_product, self.train_ratio) # Clean two versions of sales data with one aggregated and the other segmented by product
		self.df_weather = Aux.clean_weather_data(self.df_weather)

	def shutdown(self):
		'''Terminate temp objects as well as back up files'''
		self.df_weather.to_csv(self.temp_path + 'weather_data.csv')
		self.df_sales.to_csv(self.temp_path + 'sales_data.csv') # Export cleaned data to memory as backup files
		self.df_sales_product_train.to_csv(self.temp_path + 'sales_data_by_product_stacked_train.csv', index = False)
		print('{0:*^80}'.format('Cleaned Data Exported as ' + 'sales_data_by_product_stacked_train.csv'))
		self.df_sales_product_test.to_csv(self.temp_path + 'sales_data_by_product_stacked_test.csv', index = False)
		print('{0:*^80}'.format('Cleaned Data Exported as ' + 'sales_data_by_product_stacked_test.csv'))
		print('{0:*^80}'.format('Data Cleaning Process Complete'))

	def backup_import(self):
		'''Import local copies of cleaned data'''
		print('{0:*^80}'.format('Warning: Developer Mode Initiated'))
		self.df_weather = pd.read_csv(self.temp_path + 'weather_data.csv', index_col = 'date')
		self.df_sales = pd.read_csv(self.temp_path + 'sales_data.csv', index_col = ['date', 'store'])
		self.df_sales_product_train = pd.read_csv(self.temp_path + 'sales_data_by_product_stacked_train.csv', index_col = None)
		self.df_sales_product_test = pd.read_csv(self.temp_path + 'sales_data_by_product_stacked_test.csv', index_col = None)
		print('{0:*^80}'.format('Backup Cleaned Data Imported'))

	def exec(self):
		if self.dev_mode:
			self.backup_import()
		else:
			self.authenticate()
			self.data_import()
			self.data_clean()
			self.shutdown()

class Aux:
	'''Auxiliary module to structure code in the main module'''
	def clean_col_names(columns):
		'''Convert all column names into lower case, delete special characters and link words with underscores'''
		return columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

	def clean_client_data(df):
		'''Preprocess client sales data for later analytics'''
		df.drop_duplicates(inplace = True) # Eliminate duplicated columns
		df.columns = Aux.clean_col_names(df.columns) # Standardise column format
		df.rename(columns = {'total_net_sales': 'sales'}, inplace = True) # Shorten variable names
		df['date'] = pd.to_datetime(df['date']) # Specify would-be index's data type
		df['store'] = df['store'].astype('category') # Specify datatype for individual stores
		df.set_index(['date', 'store'], inplace = True) # Speicfy multiindices
		df.sort_index(inplace = True) # Order grouped data chronologically
		return df

	def clean_client_data_product(df, train_ratio):
		'''Preprocess client sales data segmented by product for later analytics'''
		df.drop_duplicates(inplace = True) # Eliminate duplicated columns
		df.columns = Aux.clean_col_names(df.columns) # Standardise column format
		df.drop(columns = ['date.1', 'week_of_year'], inplace = True) # Delete duplicate columns
		df.rename(columns = {'day_of_the_week_monday_is_0': 'day_of_the_week', 'description': 'product', 'total_average_sell': 'price', 'total_net_sales': 'sales'}, inplace = True) # Shorten variable names
		df['day_of_the_week'] = (df['day_of_the_week'].astype(int) + 1).astype('category') # Convert zero indexed columns to standard one indexed column
		df[['store', 'product', 'day_of_month', 'month']] = df[['store', 'product', 'day_of_month', 'month']].astype('category')
		df['date'] = pd.to_datetime(df['date']) # Specify data types
		df.set_index(['date', 'store', 'product'], inplace = True) # Specify multiindices
		df.sort_index(inplace = True) # Order grouped data chronologically
		df = df.loc[df['sales'] >= 0.1,:]
		#df.loc[:, df.columns.str.startswith('sales')] = df.loc[:, df.columns.str.startswith('sales')].fillna(value = 0)
		df.reset_index(inplace = True)
		train, test = df.iloc[:round(len(df) * train_ratio), :], df.iloc[round(len(df) * train_ratio):, :] # Split cleaned data into two dataframes to be exported as two files
		return train, test

	def clean_weather_data(df):
		'''Preprocess weather data for later analytics'''
		df.columns = Aux.clean_col_names(df.columns) # Standardise column format
		df['date'] = pd.to_datetime(df['date'])
		df.set_index('date', inplace = True) # Specify datatime index
		df = df.astype(float) # Specify all column datatypes
		return df

if __name__ == '__main__':
	obj = Data(config.input_path, config.temp_path, config.output_path, config.weather_url, config.dbx_token, config.dropbox_path, config.input_f_names, config.input_w_name, config.web_source)
	obj.exec()
