'''
This script contains functions that are required to be independent to avoid circular import issues
'''

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
