# Zillow Regression Project

### Executive Summary
I was able to create a regression model that beat out the baseline for predicting homes in the california area. It preformed with an R^2 value of 0.3859 and an RMSE on out of sample data of 294674.89

##### Top Predictors:
- Square Footage
- Number of bathrooms
- Number of bedrooms
##### Model Information:
- Type: Linear regression
- Alpha = 1
- RMSE: 294683.58
- R^2: 0.3859
##### Background Info:
- Single unit properties
- Data gathered was from 2017
- Counties: Los Angeles, Orange, Ventura
- Only entries with a latitude and longitude
##### Our main predictors for house prices


##### Linear Regression Model vs Baseline


## Project Description
- I will be running zillow data through the data science pipeline.*****************

## Project Goals
******************

## Buisness Goals
************

## Audience
- Data science team.

## Project Deliverables
********************

## Project Context
- Im using the zillow dataset from the codeup sql database, I am using the 2017 dataset, and filtering for the ("hot") months for selling between "2017-05-01" and "2017-08-31", and I am also selection for single-unit properties.
- This dataset originally contained 62 columns and 38619 rows after all tables were joined together.
- after cleaning the dataset we are left with 7 columns and 37928

## Data Dictionary
|Target|Datatype|Definition|
|:-------|:--------|:----------|
|tax_value|dtype('Float64')|Gives the home value after tax|

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
|sqft|dtype('Float64')|The total square footage of the house|
|bathrooms|dtype('Float64')|The number of bathrooms|
|bedrooms|dtype('Float64')|The number of bedrooms|
|year_built|dtype('Float64')|The year the house was built|
|tax_amount|dtype('Float64')|The amount that was paid in tax|
|fips|dtype('Float64')|Indentifies geographic areas|
|lot_size_sqft|dtype('Float64')|Gives total square footage of the lot|

## Hypotheses
### Alpha
- α = .05

******************
## Data Science Pipeline
#### Planning
- Make a README.md that will hold all of the project details including a data dictionary, key finding, initial hypotheses, and explain how my process can be replicated
- Create a MVP, originally and work through the iterative process of making improvements to that MVP.
- Make hypotheses that are tested though statistical analysis.
- Create visualizations throughout the process both in the explore stage and visualizing my findings after modeling.
#### Acquire
- Create an acquire.py that will take the data from sql and put it into a pandas dataframe. I saved the zillow data to a .csv for easier access
#### Prepare
- Create a prepare.py that will clean and remove outliers from the data.
- While cleaning the data I removed any outliers that fell far outside of the expected for square footage, bathrooms, bedrooms, and tax value.
#### Explore
- Awnser my initial hypotheses that was asked in my planning phase, and test those hypotheses using statistical tests, either accepting or rejecting the null hypothesis.
- Continue using statistical testing and visualizations to discover variable relationships in the data, and attempt to understand "how the data works".
- Summarize my conclusions giving clear awnsers to the questions I posed in the planning stage and summarize any takeaways that might be useful.
#### Modeling and Evaluation
- Train and evaluate  models comparing those models to the baseline on different evaluation metrics, but focusing on root mean squared error.
- Validate the models and choose the best model that was found in the validation phase.
- Test the best model found and summarize the performance and document the results, and visualize those results.
#### Delivery
************************

## Modules

#### acquire.py
- Acquires the data from the CodeUp SQL database and puts the table into a pandas dataframe
#### prepare.py
- Cleans my data and gets it ready for use in modeling and explore.
#### wrangle.py
- Combines my acquire and prepare into one easy to call function.
#### explore.py
- Contains functions that I used to help visualize the data.

## Project Reproduction
- Random state or seed = **174**, and is used in my models and my split functions.
- In replication making use of the user defined function, in cunjunction with my documented process, and presaved models should give a good guide.The functions that will make the process faster.
- Create and use your own env file that connects you to the sql database.
- Run the clustering_zillow_final jupyter notebook with all the .py files.

## Conclusion

### Key takeaways

### Model takeaways

### Moving forward

##### If given more time

