import Kami

Kami.Preprocess('/Users/Chris/Dropbox/Nusa_Kitchen_Raw_Data/nusa_data_grouped_products.csv', 'cache/')
Kami.Analyse('output/', 'cache/', weekly_agg = True, sales_as_label = True, n_sample = 100000)
Kami.Vis('output/', 'cache/')
