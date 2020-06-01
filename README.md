## Package Description

The package contains one object **Kami** with four methods in the order of execution:  
1. **Preprocess()**
2. **Analyse(n_sample)**
3. **Vis()**
4. **Forecast(store_list, product_list, start, end)**

To initiate the object **Kami**, the user is required to supply at least these three arguments:  
1. Path to the grouped product sales input data (***input_f_path***)
2. Path to an intermediary folder to store intermediary data (***cache_dir_path***)
3. Path to an output folder to store final predictions (***output_dir_path***)

While **Preprocess** and **Vis** methods are executed without any argument, **Analyse** method can be supplied with an optional argument *n_sample* which is the number of random samples drawn from the predefined training data.

**Forecast** method is required to be supplied with four arguments including:  
1. A list of stores whose sales are predicted (***store_list***)
2. A list of products whose sales are predicted (***product_list***)
3. The start date of the forecast (***start***)
4. The end date of the forecast (***end***)

## Typical Use Case

***
	from Kami import Kami

	obj = Kami(input_f_path = 'PATH_TO_SALES_DATA/SALES_DATA.csv',
			output_dir_path = 'OUTPUT_FOLDER/',
			cache_dir_path = 'CACHE_FOLDER/')
	obj.Preprocess()
	obj.Analyse()
	obj.Vis()
	obj.Forecast(store_list = ['STORE_A', 'STORE_B'],
			product_list = ['PRODUCT_A', 'PRODUCT_B'],
			start = 'MM/DD/YYYY',
			end = 'MM/DD/YYYY')
***

*The package's structure is inspired by Kaggle Rossmann sales forecast comptition third place winner Neokami whose original GitHub repository is as followed:  *
[Neokami](https://github.com/entron/entity-embedding-rossmann/tree/kaggle)
