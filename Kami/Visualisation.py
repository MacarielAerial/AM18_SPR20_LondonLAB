'''
The script creates predicted vs. actual visualisations after the neural network model is fitted
'''

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Vis:
	'''Main module'''
	def __init__(self, output_dir_path, cache_dir_path, sub_dir = 'Predicted_vs_Actual_Plots/'):
		print('{0:*^80}'.format('Predicted vs. Actual Plotting in Progress...'))
		self.configure(output_dir_path, sub_dir = sub_dir)
		merged, product_dict = self.preprocess(cache_dir_path, output_dir_path)
		self.plot(merged, product_dict, 'overall_sales', output_dir_path, sub_dir, plot_total_sales = True)
		[self.plot(merged, product_dict, item, output_dir_path, sub_dir, plot_total_sales = False) for item in product_dict]
		print('{0:*^80}'.format('Predicted vs. Actual Plotting Completed'))

	def configure(self, output_dir_path, sub_dir):
		'''Configure settings'''
		from pandas.plotting import register_matplotlib_converters
		register_matplotlib_converters()
		plt.style.use('ggplot')
		if not os.path.exists(output_dir_path + sub_dir):
			os.makedirs(output_dir_path + sub_dir)

	def preprocess(self, cache_dir_path, output_dir_path):
		'''Preprocess data for plotting'''
		test = pd.read_csv(cache_dir_path + 'test.csv', index_col = False)
		test_predicted = pd.read_csv(output_dir_path + 'test_predicted.csv', index_col = False)
		print(test['product'].value_counts()[:50])
		merged = pd.DataFrame({'date': test['date'],
					 'store': test['store'],
					 'store_idx': test_predicted['store'],
					 'product': test['product'],
					 'product_idx': test_predicted['product'],
					 'actual': test['sales'],
					 'predicted': test_predicted['predicted']})
		merged['date'] = pd.to_datetime(merged['date'], format = '%Y-%m-%d')
		product_dict = dict(zip(merged['product'].array, merged['product_idx'].array))
		return merged, product_dict

	def plot(self, merged, product_dict, item, output_dir_path, sub_dir, plot_total_sales):
		'''Plot actual vs. predicted plots for all products and overall sales'''
		if plot_total_sales:
			data = merged.groupby('date').agg({'predicted': 'sum', 'actual': 'sum'}).sort_index(ascending = True)
		else:
			data = merged.loc[merged['product'] == item, :].groupby('date').agg({'predicted': 'sum', 'actual': 'sum'}).sort_index(ascending = True)
		plt.figure(figsize = (16, 9))
		plt.plot(data['actual'], color = 'red', label = 'Actual')
		plt.plot(data['predicted'], color = 'blue', label = 'Predicted')
		plt.title('Predicted vs. Actual for Neokami Model')
		plt.ylabel('Target Variable')
		plt.xlabel('Date')
		plt.title(item)
		plt.legend()
		plt.savefig(output_dir_path + sub_dir + item.replace(' ', '_').replace('/', '_').lower() + '.png', dpi = 300)
		plt.close()
