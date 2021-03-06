
# Beating the Zestimate

<!-- Add banner here -->
![Banner](https://webassets.inman.com/wp-content/uploads/2014/03/zillow-logo-use-this-one.png)

# Zillow Clustering Project

<!-- Add buttons here -->

![GitHub release (latest by date including pre-releases)](https://img.shields.io/badge/release-draft-yellow)
![GitHub last commit](https://img.shields.io/badge/last%20commit-Sep%202021-green)

<!-- Describe your project in brief -->
The Zestimate is a powerful tool used by zillow to predict the final sale price of realestate. My goal for this project is to identify whats driving error between the Zestimate and the final sale prices. To accomplish this goal I will be utilizing clustering and regression models. I will present my findings via a notebook walkthrough to my datascience team. 



# Executive Summary
<!-- Add a demo for your project -->

### Goals
- Work through datascience pipline for zillow data, documenting the process.
- Use clustering and regression modeling to find clusters of houses that will imporove our model.
- Find drivers for log error in the zestimate.

### Statistical testing:
I found that Location bedrooms and bathrooms all have a relationship with log error.

I also found that the log error was different when comparing large units to small units, and when comparing large lots to small lots.
### Clustering:
- The cluster of longitude and latited was not effective in predicting logerror
- The cluster of longitude latitude and square footage was not effective in predicting logerror
- The cluster of bedroom bathroom and tax value was not effective in predicting logerror

I cannot accuratly predict the log error in this amount of time, my model came just short of the baseline. I was able to find some key incites about the data including several statistically significant finding on variables relationships to log error.

Model features: **bathrooms, bedrooms, square_footage, house_basics_cluster, loc_sqft_cluster**

**My models performance was almost identical to the baseline**

### Takeaways/keypoints
- It seems like location plays a factor some locations might have inflated or deflated housing markets.
- The zestimate seems to slightly overvalue smaller houses/lots while undervalueing larger houses/lots.
- log error is slightly higher in units that have 5-6 bedrooms or bathrooms.

### Recommendations
The zestimate team should focus on improving the zestimate using some of the drivers I outlined, and try to further encorperate these values into the zestimate moving forward.

# Table of contents
<!-- Add a table of contents for your project -->

- [Project Title](#project-title)
- [Executive Summary](#executive-summary)
- [Table of contents](#table-of-contents)
- [Data Dictionary](#data-dictionary)
- [Data Science Pipeline](#data-science-pipline)
    - [Acquire](#acquire)
    - [Prepare](#prepare)
    - [Explore](#explore)
    - [Model](#model)
    - [Evaluate](#evaluate)
- [Conclusion](#conclusion)
- [Given More Time](#given-more-time)
- [Recreate This Project](#recreate-this-project)
- [Footer](#footer)

# Data Dictionary
[(Back to top)](#table-of-contents)
<!-- Drop that sweet sweet dictionary here-->

| Feature                    | Datatype                | Definition   |
|:---------------------------|:------------------------|:-------------|
| parcelid| 55513 non-null: int64|individual id for unique properties|
| baths| 55513 non-null: float64|# of bathrooms a property has|
| beds| 55513 non-null: float64|# of bedrooms a property has|
| sqft| 55513 non-null: float64|calculated square footage of home|
| latitude| 55513 non-null: float64 |where the porperty is located in refrence to latitude|
| longitude| 55513 non-null: float64 |where the porperty is located in refrence to  longitude|
| lotsizesquarefeet| 55513 non-null: float64 |the square footage of the land the propety resides on|
| regionidcity| 55513 non-null: object|unique identifier for cities the property is in|
| regionidzip| 55513 non-null: object|uniques identifier for the zip code the propert resides in|
| year_built| 55513 non-null: float64|year the property was built|
| structuretaxvaluedollarcnt| 55513 non-null: float64 |the estimated tax value of the property itself|
| tax_value| 55513 non-null: float64 |the estimated tax value of the property|
| landtaxvaluedollarcnt| 55513 non-null: float64 |the estimated tax value of the land the property is on|
| tax_amount| 55513 non-null: float64 |how much the owner of the property must pay this year|
| logerror| 55513 non-null: float64 |the target of this project (error produced in predictions)|
| transactiondate| 55513 non-null: object  |date property was sold|
| propertylandusedesc| 55513 non-null: object  |what the property is listed as ex.(Single family)|
| LA| 55513 non-null: uint8|whether or not the propert resides in LA county|
| Orange| 55513 non-null: uint8|whether or not the propert resides in Orange county|
| Ventura| 55513 non-null: uint8|whether or not the propert resides in Ventura county|
| county| 55513 non-null: object|The county the resident resides in|
| taxrate| 55513 non-null: float64 |gives the tax rate for each property|
| structure_dollar_per_sqft| 55513 non-null: float64 |gives the structures cost per square foot|
| land_dollar_per_sqft| 55513 non-null: float64 |gives the land cost per square foot|


# Data Science Pipeline
[(Back to top)](#table-of-contents)
<!-- Describe your Data Science Pipeline process -->
Following best practices I documented my progress throughout the project and will provide quick summaries and thoughts here. For a further deep dive visit my (enter explore notebook here) & (enter final notebook here)

### Acquire
[(Back to top)](#table-of-contents)
<!-- Describe your acquire process -->
The data was acquired from the Codeup MySQL server using the zillow database. I pulled every property from the properties_2017 table (later in prepare I will filter this down further) and joined the following tables:

- airconditioningtype (for labeling purposes)
- architecturalstyletype (for labeling purposes)
- buildingclasstype (for labeling purposes)
- heatingorsystemtype (for labeling purposes)
- predictions_2017 (for logerror which will be our target)(( I also filtered by transaction date and parcel id to handle duplicates))

My goal with this acquisition was to give me as much data as possible moving foward.
At this point our data has
* *77614 rows*
* *74 columns*

This can all be found in my acquire.py file in my github

### Prepare
[(Back to top)](#table-of-contents)
<!-- Describe your prepare process -->
Performed the following on my acquired data.

- dropped null values from columns and rows which had less than 50% of the values.
- dropped all data from properties that where not single value homes
    - I quantified single family homes as properties with a propertylandusetypeid of:
        - 261	Single Family Residential
        - 262	Rural Residence
        - 263	Mobile Home
        - 264	Townhouse
        - 265	Cluster Home
        - 268	Row House
        - 273	Bungalow
        - 275	Manufactured, Modular, Prefabricated Homes
        - 276	Patio Home
        - 279	Inferred Single Family Residential
- dropped the duplicated columns pulled over from the sql inquiry
- removed outliers by upper and lower iqr fences from
    - calculatedfinishedsquarefeet
    - bedroomcnt
    - bathroomcnt
    - taxvaluedollarcnt
    - calculatedfinishedsquarefeet
- Further removed outliers manually with the following conditions
    - bathroom count or bedroom count greater than 7
    - bathroom coutnt or bedroom count less than 1 
    - properties with greater than 200000 square feet for lot size
    - properties with a square footage above 8,000
- Drops columns that have no use
    - id because its a usless and duplicated
    - heatingorsystemtypeid because it was missing about 20k values to much to fill
    - heatingorsystemdesc because it was missing about 20k values to much to fill
    - propertylandusetypeid is useless to me after the dropping irrelevant data earlier
    - buildingqualitytypeid because it was missing about 20k values to much to fill
    - rawcensustractandblock useless data to me
    - unitcnt is useless to me after the dropping irrelevant data earlier
    - propertyzoningdesc because it was missing about 20k values to much to fill
    - censustractandblock isn't useful to me
    - calculatedbathnbr data is inconsistent 
    - finishedsquarefeet12 calculatedsquarefeet is a better metric
    - fullbathcnt redundant to bathroom count
    - assessmentyear values are all 2016
    - propertylandusetypeid because the data was filtered already. 
    - roomcnt because it is inconsistent with data
    - county because we already created dummy variables to enumerate it.
- Created boolean columns for county
- Replace fips with county column for exploration purposes.
- Removed lot sizes of over 200000, when working witht he data I found these values suspicious and dont believe they will be useful.
- Filled null values in the following columns
    - year built (filled nulls with 2017)
    - lotsizesquarefeet (filled nulls with median ie. 7313)
    - buildingqualitytypeid (filled nulls with most common ie. 6)
- Dropped remaining null values
- Created a column for tax value that gives a percentage tax rate on the property
- Created a column for structure cost per square foot
- Created a column for land cost per square foot
- Renamed several columns for readability.

At this point our data has
* *57889 rows*
* *25 columns*

We will now split our data into train, validate, and split.
- Split the data into 60% train, 20% validate, 20% test


### Explore
[(Back to top)](#table-of-contents)
<!-- Describe your explore process -->

- Visualized using both univariate analysis and bivariate
- Ran five statistical test comparing different variables and log error
- Created three different clusters 
    - Cluster 1: uses longitude and latitude
    - Cluster 2: uses longitude latitude and square footage
    - Cluster 3: uses bedroom bathroom and tax value

### Model
[(Back to top)](#table-of-contents)
<!-- Describe your modeling process -->
Model types: linear regression, lasso lars, tweedie, polynomial 3rd degree
**Winning model: linear regression**
Created 4 different models using bathrooms,bedrooms,square_footage,cluster 2, and cluster 3 as features.
Baseline RMSE on test data: 0.1656757
**My models RMSE on test data: 0.1656116**

### Evaluate
[(Back to top)](#table-of-contents)
<!-- Describe your evaluation process -->
I gathered the RMSE and R^2 for all of my models and put them into dataframes for easy comparison.

# Conclusion
[(Back to top)](#table-of-contents)
<!-- Wrap up with conclusions and takeaways -->
- We were able to find some indicators of of log error but will definetly need more time to create a better model.
- the model beat the baseline by a slim margin.
- Both my model and the baseline had an **RMSE of ~.1656**

# Given More Time
[(Back to top)](#table-of-contents)
<!-- LET THEM KNOW WHAT YOU WISH YOU COULD HAVE DONE-->
- I could spend 1000 hours and still continue learning new things about this data.
- I would try to make clusters that target on a neighborhood level, seeking values that the homes on the same street sold for.
- I would also improve on my models using my current model as the new baseline.
# Recreate This Project
[(Back to top)](#table-of-contents)
<!-- How can they do what you do?-->
**To Recreate This Project**
- Create your own username and password in order to access the Codeup SQL database.
- I created a wrangle file that will do all of the acquir/prepare stage.
- Use the seed 174
- Importing all of my modules and utilizing them will help expadite the process.
# Footer
[(Back to top)](#table-of-contents)
<!-- LET THEM KNOW WHO YOU ARE (linkedin links) close with a joke. -->
You can find the rest of my work here: https://github.com/JaredVahle
Add me on Linkedin here: https://www.linkedin.com/in/jared-vahle-data-science/
Check out my Trello board here: https://trello.com/b/nvbynjZ6/clustering-zillow-data
