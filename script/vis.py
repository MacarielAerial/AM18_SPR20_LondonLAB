import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

test = pd.read_csv(config.temp_path + 'sales_data_by_product_stacked_test.csv')
print(test['product'].value_counts())

actual = pd.read_csv(config.output_path + 'actual_test_data.csv')
predicted = pd.read_csv(config.output_path + 'predicted_test_data.csv')
predicted['actual'] = actual['actual']
predicted['date'] = pd.to_datetime(actual['date'], format = '%Y-%m-%d')
predicted['product_name'] = test['product']
predicted.to_csv(config.output_path + 'evaluation.csv', index = False)
menu_dict = dict(zip(predicted['product_name'].array, predicted['product'].array))

predicted_agg = predicted.groupby('date').agg({'predicted': 'sum', 'actual': 'sum'}).sort_index(ascending = False)

def plot(predicted_agg, item, suffix, total_sales = False):
	if not total_sales:
		predicted_agg = predicted.loc[predicted['product'] == menu_dict[item], :].groupby('date').agg({'predicted': 'sum', 'actual': 'sum'}).sort_index(ascending = False)
	plt.figure(figsize = (16, 9))
	plt.plot(predicted_agg['actual'], color = 'red', label = 'Actual')
	plt.plot(predicted_agg['predicted'], color = 'blue', label = 'Predicted')
	plt.title('Predicted vs. Actual for Neokami Model')
	plt.ylabel('Sales')
	plt.xlabel('Date')
	plt.title(item)
	plt.legend()
	plt.savefig(config.output_path + 'neokami/Neokami_' + suffix + '.png', dpi = 300)
	plt.close()

plot(predicted_agg, 'Overall Sales', suffix = 'overall', total_sales = True)

for item in menu_dict:
	plot(predicted_agg, item, suffix = item.replace(' ', '_').replace('/', '_').lower())
