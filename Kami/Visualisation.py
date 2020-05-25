'''
The script creates predicted vs. actual visualisations after the neural network model is fitted
'''

# Import libraries
import os
import pickle
import numpy as np
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt

class Vis:
	'''Main module'''
	def __init__(self, output_dir_path, cache_dir_path, sub_dir = 'Predicted_vs_Actual_Plots/'):
		self.merged = pd.DataFrame()
		self.product_dict = {}
		print('{0:*^80}'.format('Predicted vs. Actual Plotting in Progress...'))
		self.configure(output_dir_path, sub_dir = sub_dir)
		self.preprocess(cache_dir_path, output_dir_path)
		self.plot_predicted_vs_actual(self.merged, self.product_dict, 'overall_sales', output_dir_path, sub_dir, plot_total_sales = True)
		[self.plot_predicted_vs_actual(self.merged, self.product_dict, item, output_dir_path, sub_dir, plot_total_sales = False) for item in self.product_dict]
		print('{0:*^80}'.format('Predicted vs. Actual Plotting Completed'))
		Vis2(output_dir_path, cache_dir_path)
		print('{0:*^80}'.format('Product Embedding Plotting Completed'))		

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
		test_predicted = pd.read_csv(cache_dir_path + 'test_predicted.csv', index_col = False)
		print(test['product'].value_counts()[:50])
		self.merged = pd.DataFrame({'date': test['date'],
					 'store': test['store'],
					 'store_idx': test_predicted['store'],
					 'product': test['product'],
					 'product_idx': test_predicted['product'],
					 'actual': test['sales'],
					 'predicted': test_predicted['predicted']})
		self.merged['date'] = pd.to_datetime(self.merged['date'], format = '%Y-%m-%d')
		self.merged.to_csv(output_dir_path + 'evaluation.csv', index = False)
		self.product_dict = dict(zip(self.merged['product'].array, self.merged['product_idx'].array))

	def plot_predicted_vs_actual(self, merged, product_dict, item, output_dir_path, sub_dir, plot_total_sales):
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

class Vis2:
	'''Secondary module'''
	def __init__(self, output_dir_path, cache_dir_path):
		self.load_embeddings(cache_dir_path)
		self.load_label_encoders(cache_dir_path)
		self.plot_product_embeddings(output_dir_path)

	def load_embeddings(self, cache_dir_path):
		with open(cache_dir_path + 'embeddings.pickle', 'rb') as f:
			self.store_embedding, self.product_embedding, self.dow_embedding, self.dom_embedding, self.year_embedding, self.month_embedding = pickle.load(f)

	def load_label_encoders(self, cache_dir_path):
		with open(cache_dir_path + 'les.pickle', 'rb') as f:
			les = pickle.load(f)
		self.le_store, self.le_product, self.le_dow, self.le_dom, self.le_year, self.le_month = les[0], les[1], les[2], les[3], les[4], les[5]

	def plot_product_embeddings(self, output_dir_path):
		tsne = manifold.TSNE(init = 'pca', random_state = 0, method = 'exact', perplexity = 5, learning_rate = 100)
		Y = tsne.fit_transform(self.product_embedding)
		fig, ax = plt.subplots(figsize = (96, 54))
		ax.scatter(-Y[:, 0], -Y[:, 1])
		text = [ax.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontfamily = 'monospace', fontsize = 6) for i, txt in enumerate(self.le_product.classes_)]
		ax.set_title('Product Embedding Plot')
		fig.savefig(output_dir_path + 'product_embedding.pdf')
		plt.close(fig)

