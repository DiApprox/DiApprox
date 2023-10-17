# DiApprox: Differential Privacy-based Online Range Queries Approximation for Multidimensional Data
This code is related to the work presented in ACM SAC 2024

## Requirements:
1. Python (3.7)
2. Postgres

## Dependencies installations
Run the following command : `./DiApprox/install_dependencies.sh`

## Multi-objectives problem analysis
Run the following command : `./DiApprox/lunch_op_analysis.sh`

## Reel datasets and Scalability evaluations

### Datasets:
All the datasets are included in this repository except of 'Amazon Review' due to its size (146GB). To test \textit{DiApprox} on this dataset you need to follow this steps :
1. Download the dataset ([docs](https://nijianmo.github.io/amazon/index.html)) from here : [file](https://nijianmo.github.io/amazon/index.html#complete-data) 
2. Place the file in ./DiApprox/src/Data/Amazon
3. Run the script : `python3 ./DiApprox/src/Data/Amazon/processing.py`

### Database
In order to connect DiApprox to your local instance of postgres, you need to edit `./DiApprox/src/Connection.py` :
1. Database schema name you created locally [tuto] (https://medium.com/coding-blocks/creating-user-database-and-adding-access-on-postgresql-8bfcd2f4a91e)
2. The logins to postgres {user, password}

### run the experiments
Run the following command : `./DiApprox/lunch_tests.sh`





