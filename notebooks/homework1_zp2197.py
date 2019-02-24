#!/usr/bin/env python
# coding: utf-8

# # COMSW4995_007_2018_3 Elements of Data Science 
# # Homework 1
# 
# ### Due: 9pm Oct. 4
# 
# In this homework we practice loading and transforming data.  
# We'll also practice calculating summary statistics and a few visualizations.
# 
# ## Instructions
# 
# Follow the comments below and fill in the blanks (\_\_\_\_) to complete.

# ---

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# tell jupyter to display images in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# for pretty printing
import pprint

# To suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# To increase the number of rows pandas will print before truncation
pd.set_option('display.max_rows', 500)

# To increase the number of columns pandas will print before truncation
pd.set_option('display.max_columns', 100)

# To increase the width of each column pandas will print before truncating.
pd.set_option('display.max_colwidth',100)


# ---

# ## Part 1: College Football Dataset and total_score
# 
# Fall is college football season, so we'll first look at some data from the 2017 college football season.  
# This data was gathered from 'http://sports.snoozle.net/'.
# 
# We'd like to find the total score for each game and the min, max and mean of the total scores.
# 
# In this dataset, each row contains information about a single game, visiting team info followed by home team.  
# We'll need to combine two columns to find the total number of points.
# 
# This csv contains columns with duplicate names.
# When there are columns with identical names, pandas attaches a suffix to discriminate them.  
# All 'Home Team' columns will have a '.1' suffix.

# In[3]:


# Load the data from ../data/cfb_2017.csv into the variable df_cfb
df_cfb = pd.read_csv("../data/cfb_2017.csv")
df_cfb


# In[4]:


# Print the first 3 rows of the dataset using head
df_cfb.head(3)


# In[5]:


# print out information about the dataframe using .info()
df_cfb.info()


# In[6]:


# How may records are in the dataset?
print('dataframe has {} records'.format(len(df_cfb)))


# In[7]:


# How many values are missing from the Score and Score.1 columns?
print('there are {} missing scores'.format(len(df_cfb[df_cfb.Score.isnull()]) + len(df_cfb[df_cfb.loc[:,'Score.1'].isnull()])))


# In[8]:


# What is the average (mean) score for visiting teams?
print('the mean visiting team score is {:0.2f}'.format(np.mean(df_cfb.Score)))


# In[9]:


# To examine the total score, we need to combine Score and Score.1
# Create a new column called 'total_score' which is the sum of the 'Score' and 'Score.1' columns
df_cfb['total_score'] = df_cfb.Score + df_cfb.loc[:,'Score.1']
df_cfb


# In[10]:


# What is the average (mean) total score?
print('the mean total score is {:0.2f}'.format(df_cfb.total_score.mean()))


# In[11]:


# Use .describe to show other values associated with total score
df_cfb.describe().loc[:, ['Score','Score.1','total_score']]


# In[12]:


# what are the min and max values for total_score?
print('the minimum and maximum values are {} and {}'.format(df_cfb.total_score.min(), df_cfb.total_score.max()))


# In[13]:


# What were the team names and team scores for the game with the highest total score?
# We should see a single row with the columns: Vis Team, Score, Home Team, Score.1
#df_cfb[df_cfb.total_score == np.max(df_cfb.total_score)].loc[:,['Vis Team', 'Score', 'Home Team', 'Score.1']]
df_cfb[df_cfb.total_score == df_cfb.total_score.max()].loc[:,['Vis Team', 'Score', 'Home Team', 'Score.1']]


# In[14]:


# Use seaborn distplot to plot the distribution of total_score
# Turn on rug to show each game's total score
_ = sns.distplot(df_cfb.total_score, rug = True)


# ---

# ## Part 2: World Bank Data
# 
# This data is provided by World Bank Open Data https://data.worldbank.org/.
# 
# It includes many country data indicators sampled over time.
# 
# There are two files we are interested in:
# 1. WDICountry.csv includes country and region information, one country or region per row.
# 2. WDIData.csv includes indicator data, one row per country and indicator.
# 
# We would like to be able to analyze a few indicators for countries grouped by region.  
# To do that we will need to clean and join the two sets of records.

# ---

# ### Part 2a: Munge WDICountry

# In[15]:


# Read Country information from '../data/WDICountry.csv' into df_country
# print the number of rows in df_country
df_country = pd.read_csv('../data/WDICountry.csv')
print('df_country has {} rows'.format(len(df_country)))


# In[16]:


# Print the first 3 rows of WDICountry
df_country.head(3)


# In[17]:


# Using .columns, how many columns does WDICountry have?
print('df_country has {} columns'.format(df_country.columns.size))


# In[18]:


# We'll only keep a few columns: ['Country Code','Short Name','Region','Income Group']
# Overwrite df_country with a new dataframe containing only these columns
# Print out the statement 'df_country has {} columns' using .format to confirm that there are only 4 columns
df_country = df_country.loc[:,['Country Code','Short Name','Region','Income Group']]
print('df_country has {} columns'.format(df_country.columns.size))


# In[19]:


# Examine df_country using .info
df_country.info()


# In[20]:


# There are some rows with missing Region and Income Group.
# Print out both the number and proportion of rows with missing Region information
n_missing = len(df_country[df_country.Region.isnull()])
prop_missing = n_missing / len(df_country)
print('there are {:} rows with missing data, {:0.2} of the dataset'.format(n_missing,prop_missing))


# In[21]:


# Drop the rows of df_country with any null values (using inplace=True)
# Use .info to make sure there a no longer null values
df_country.dropna(axis = 0, inplace = True)
df_country.info()


# In[22]:


# Each row of df_country should be a separate country
# Assert that there are no duplicates (use len and drop_duplicates)
# hint: df_country should be the same length before and after dropping duplicate rows
assert len(df_country) == len(df_country.drop_duplicates())


# In[23]:


# Assert that 'Country Code' is unique (use len and unique)
# hist: the number of unique country codes should be the same length as the dataframe
assert len(df_country) == len(df_country['Country Code'].unique())


# In[24]:


# Set the index (.set_index) of df_country to be 'Country Code' (inplace) and display the first 5 rows
df_country.set_index('Country Code',inplace = True)
df_country.head()


# ---

# ### Part 2b: Munge WDIData

# In[25]:


# Now we'll load the other country data we're interested in.
# Read csv '../data/WDIData.csv.zip' into df_data
# Note: 
#  Since this file is large it is stored as a zip.
#  You don't need to decompress the zip file first, pandas will handle this for you.
df_data = pd.read_csv('../data/WDIData.csv.zip')


# In[26]:


# Display .info for df_data
# Note that the data is in long format instead of wide format
df_data.info()  # (401016, 62)


# In[27]:


# Use pprint.pprint to print a list of the unique values in Indicator Name
# These are all of the available data points in this file, which is why it's so large.
# (to see the difference, try using the standard 'print' first)
#print(df_data['Indicator Name'].unique().tolist())
pprint.pprint(df_data['Indicator Name'].unique().tolist())


# In[28]:


# We'll only keep a few of these indicators.
# Create a list of indicators to keep that includes these indicators:
# 'Employment to population ratio, 15+, female (%) (modeled ILO estimate)','GDP (constant 2010 US$)','Population, total','Population density (people per sq. km of land area)','Unemployment, total (% of total labor force) (national estimate)'
data_indicators_to_keep = ['Employment to population ratio, 15+, female (%) (modeled ILO estimate)','GDP (constant 2010 US$)','Population, total','Population density (people per sq. km of land area)','Unemployment, total (% of total labor force) (national estimate)']


# In[29]:


# The columns of WIData contain information for each year.
# We'll look at only year 2016
# Create a list of columns to keep that includes these columns:
# 'Country Code','Indicator Name','2016'
data_columns_to_keep = ['Country Code','Indicator Name','2016']


# In[30]:


# Overwrite df_data witha dataframe containing only: 
#  the rows whose 'Indicator Name' is in data_indicators_to_keep (use .isin) and 
#  the columns in data_columns_to_keep
df_data = df_data[df_data['Indicator Name'].isin(data_indicators_to_keep)].loc[:,data_columns_to_keep]


# In[31]:


# Display the first 5 rows of df_data
df_data.head()


# In[32]:


# Into df_data_pivot, .pivot df_data with index 'Country Code', columns 'Indicator Name' and values '2016'
# Display the first 5 rows of df_data_pivot
df_data_pivot = df_data.pivot(index = 'Country Code',
                             columns = 'Indicator Name',
                             values = '2016')
df_data_pivot.head()


# ---

# ### Part 2c: Join the two datasets

# In[33]:


# Importantly, df_country and df_data_pivot now have the the same index values.
# To see this, use .index to print out the first 5 index values in df_country and df_data
print(df_country.index)
print(df_data_pivot.index)


# In[34]:


# Into df_wdi put the inner join of df_country with df_data_pivot using .join
# Display the first 5 rows
df_wdi = df_country.join(df_data_pivot, how = 'inner')
df_wdi.head()


# In[35]:


# Assert that the number of rows matches the number of unique 'Short Name'
assert len(df_wdi) == len(df_wdi['Short Name'].unique())


# ---

# ### Part 2d: Analysis and Visualization

# In[36]:


# Display the number of countries per region seen in df_wdi (use .value_counts)
df_wdi.Region.value_counts()


# In[37]:


# What proportion of our dataset is made up by each region?
# Hint: Divide the output of value_counts by the number of rows.
df_wdi.Region.value_counts() / len(df_wdi)


# In[38]:


# Display the summary stats (means and quartiles) for 'Employment to population ratio, 15+, female (%) (modeled ILO estimate)'
df_wdi.loc[:,['Employment to population ratio, 15+, female (%) (modeled ILO estimate)']].describe()


# In[39]:


# Let's rename that column to something shorter.
# Use .rename to rename (inplace)
#  'Employment to population ratio, 15+, female (%) (modeled ILO estimate)' 
#  to 
#  'Female Employment To Population Ratio'
# Display the columns to confirm.
df_wdi.rename({'Employment to population ratio, 15+, female (%) (modeled ILO estimate)':'Female Employment To Population Ratio'},
              axis = 1,
              inplace = True)
df_wdi.columns


# In[40]:


# Use seaborn .catplot to display box plots of 'Female Employment Ratio' for each Region
# Since the region names are long, we'll use horizontal box plots.
# Put 'Female Employment To Population Ratio' on the x-axis and 'Region' on the y-axis.
# Set 'aspect' to 2 to widen the plot
_ = sns.catplot(x = 'Female Employment To Population Ratio',
               y = 'Region',
               data = df_wdi,
               kind = 'box',
               aspect = 2)

