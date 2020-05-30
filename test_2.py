from Kami import Kami

obj = Kami(input_f_path = '/Users/Chris/Dropbox/Nusa_Kitchen_Raw_Data/nusa_data_grouped_products.csv',
		 output_dir_path = 'output/',
		 cache_dir_path = 'cache/')
#obj.Preprocess()
#obj.Analyse(n_sample = 5000)
#obj.Vis()
obj.Forecast(store_list = ['006 Nusa Basinghall Street', '005 New Fetter Lane'],
		product_list = ['Thai Style Crushed Chilli Chicken 16oz', 'Bombay Potato & Pea 16oz'],
		start = '05/01/2020',
		end = '05/31/2020')
