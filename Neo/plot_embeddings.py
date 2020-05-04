import pickle
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import config

class Vis():
	def __init__(self, output_path):
		self.load_embeddings(output_path)
		self.load_label_encoders(output_path)
		self.plot_store_embeddings(output_path)
		self.plot_product_embeddings(output_path)
		self.plot_dow_embeddings(output_path)
		self.plot_dom_embeddings(output_path)
		self.plot_year_embeddings(output_path)
		self.plot_month_embeddings(output_path)

	def load_embeddings(self, output_path):
		with open(output_path + 'embeddings.pickle', 'rb') as f:
			self.store_embedding, self.product_embedding, self.dow_embedding, self.dom_embedding, self.year_embedding, self.month_embedding = pickle.load(f)
	
	def load_label_encoders(self, output_path):
		with open(output_path + 'les.pickle', 'rb') as f:
			les = pickle.load(f)
		self.le_store, self.le_product, self.le_dow, self.le_dom, self.le_year, self.le_month = les[0], les[1], les[2], les[3], les[4], les[5]

	def plot_store_embeddings(self, output_path):
		tsne = manifold.TSNE(init = 'pca', random_state = 0, method = 'exact', perplexity = 5, learning_rate = 100)
		Y = tsne.fit_transform(self.store_embedding)
		fig, ax = plt.subplots(figsize = (16, 9))
		ax.scatter(-Y[:, 0], -Y[:, 1])
		text = [ax.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontsize = 8) for i, txt in enumerate(self.le_store.classes_)]
		ax.set_title('Store Embedding Plot')
		fig.savefig(output_path + 'store_embedding.pdf')
		plt.close(fig)

	def plot_product_embeddings(self, output_path):
		tsne = manifold.TSNE(init = 'pca', random_state = 0, method = 'exact', perplexity = 5, learning_rate = 100)
		Y = tsne.fit_transform(self.product_embedding)
		fig, ax = plt.subplots(figsize = (96, 54))
		ax.scatter(-Y[:, 0], -Y[:, 1])
		text = [ax.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontfamily = 'monospace', fontsize = 6) for i, txt in enumerate(self.le_product.classes_)]
		ax.set_title('Product Embedding Plot')
		fig.savefig(output_path + 'product_embedding.pdf')
		plt.close(fig)

	def plot_dow_embeddings(self, output_path):
		tsne = manifold.TSNE(init = 'pca', random_state = 0, method = 'exact', perplexity = 5, learning_rate = 100)
		Y = tsne.fit_transform(self.dow_embedding)
		fig, ax = plt.subplots(figsize = (16, 9))
		ax.scatter(-Y[:, 0], -Y[:, 1])
		text = [ax.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontsize = 8) for i, txt in enumerate(self.le_dow.classes_)]
		ax.set_title('Day of Week Embedding Plot')
		fig.savefig(output_path + 'dow_embedding.pdf')
		plt.close(fig)

	def plot_dom_embeddings(self, output_path):
		tsne = manifold.TSNE(init = 'pca', random_state = 0, method = 'exact', perplexity = 5, learning_rate = 100)
		Y = tsne.fit_transform(self.dom_embedding)
		fig, ax = plt.subplots(figsize = (16, 9))
		ax.scatter(-Y[:, 0], -Y[:, 1])
		text = [ax.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontsize = 8) for i, txt in enumerate(self.le_dom.classes_)]
		ax.set_title('Day of Month Embedding Plot')
		fig.savefig(output_path + 'dom_embedding.pdf')
		plt.close(fig)

	def plot_year_embeddings(self, output_path):
		tsne = manifold.TSNE(init = 'pca', random_state = 0, method = 'exact', perplexity = 5, learning_rate = 100)
		Y = tsne.fit_transform(self.year_embedding)
		fig, ax = plt.subplots(figsize = (16, 9))
		ax.scatter(-Y[:, 0], -Y[:, 1])
		text = [ax.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontsize = 8) for i, txt in enumerate(self.le_year.classes_)]
		ax.set_title('Year Embedding Plot')
		fig.savefig(output_path + 'year_embedding.pdf')
		plt.close(fig)

	def plot_month_embeddings(self, output_path):
		tsne = manifold.TSNE(init = 'pca', random_state = 0, method = 'exact', perplexity = 5, learning_rate = 100)
		Y = tsne.fit_transform(self.month_embedding)
		fig, ax = plt.subplots(figsize = (16, 9))
		ax.scatter(-Y[:, 0], -Y[:, 1])
		text = [ax.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', family = 'cursive', fontsize = 8) for i, txt in enumerate(self.le_month.classes_)]
		ax.set_title('Month Embedding Plot')
		fig.savefig(output_path + 'month_embedding.pdf')
		plt.close(fig)






if __name__ == '__main__':
	Vis(config.output_path)
