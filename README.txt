# AM18_SPR20_LondonLAB

The package contains two functions: Preprocess and Analyse.

Preprocess has two arguments: 1. Path to the grouped product sales input data
			      2. Path to an intermediary folder to store intermediary data

Analyse has two arguments as well: 1. Path to an output folder to store final output
				   2. Path to the intermediary folder specified during Preprocess

Vis has two arguments: 1. Path to an output folder to store plots
		       2. Path to the intermediary folder specified during Preprocess

Forecast has 6 arguments: 1. Path to an output folder to store final predictions
		        2. Path to the intermediary folder specified during Preprocess
		        3. A list of stores whose sales are predicted
		        4. A list of products whose sales are predicted
		        5. The start date of predictions
		        6. The end date of predictions

To obtain final results, call Preprocess and Analyse sequentially with their respective arguments after setting up the intermediary and the output folder
