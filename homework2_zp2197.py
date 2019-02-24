#!/usr/bin/env python
# coding: utf-8

# # Homework 2
# 
# ### Due: Sun Oct. 21 @ 9pm

# In this homework we'll perform a hypothesis test and clean some data before training a regression model.
# 
# 
# ## Instructions
# 
# Follow the comments below and fill in the blanks (____) to complete.

# In[87]:


import pandas as pd
import numpy as np
from pprint import pprint
import seaborn as sns
import sklearn
import matplotlib.pylab as plt

# To suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Part 1: Hypothesis Testing with an A/B test

# Suppose we work at a large company that is developing online data science tools. Currently the tool has interface type A but we'd like to know if using interface tool B might be more efficient.
# To measure this, we'll look at length of active work on a project (aka project length).
# We'll perform an A/B test where half of the projects will use interface A and half will use interface B.

# In[88]:


# read in project lengths from '../data/project_lengths'
# there should be 1000 observations for both interfaces
df_project = pd.read_csv('../data/project_lengths.csv')
df_project.info()


# In[89]:


# calculate the difference in mean project length between interface A and B
# for consistency, subtracting A from B
# hint: this number should be negative here (could interpret as faster)
mean_A = df_project.lengths_A.mean()
mean_B = df_project.lengths_B.mean()
observed_mean_diff = mean_B - mean_A
observed_mean_diff


# In[90]:


# we'll perform a permutation test to see how significant this result is
# generate 10000 random permutation samples of mean difference
# hint: use np.random.permutation
rand_mean_diffs = []
n_samples = 10000
combined_times = np.concatenate([df_project.lengths_A.values, df_project.lengths_B.values])
n_A = len(df_project) # number of observations for page A
for i in range(n_samples):
    rand_perm = np.random.permutation(combined_times)
    rand_mean_A = np.mean(rand_perm[:n_A])
    rand_mean_B = np.mean(rand_perm[n_A:])
    rand_mean_diffs.append(rand_mean_B - rand_mean_A)


# In[91]:


# use seaborn to plot the distribution of mean differences
# use plt.vlines to plot a line at our observed difference in means (ymin=0,ymax=0.5)
_ = sns.distplot(rand_mean_diffs, norm_hist=True, kde=False)
_ = plt.vlines(observed_mean_diff,ymin=0,ymax=0.5,color = 'r')


# In[92]:


# the plot should seem to indicate significance, but let's calculate a one-tailed p_value using rand_mean_diffs
p_value = sum(np.array(rand_mean_diffs) <= observed_mean_diff) / len(rand_mean_diffs)
p_value


# In[93]:


# we can calculate the effect size of our observation
# this is the absolute value of the observed_mean_diff divided by the standard deviation of the combined_times
observed_effect_size = np.abs(observed_mean_diff) / np.std(combined_times)
observed_effect_size


# In[94]:


# we'll use this for the next 2 steps
from statsmodels.stats.power import tt_ind_solve_power


# In[95]:


# what is the power of our current experiment?
# e.g. how likely is it that correctly decided that B is better than A 
#   given the observed effect size, number of observations and alpha level we used above
# since these are independent samples we can use tt_ind_solve_power
# hint: the power we get should not be good
power = tt_ind_solve_power(effect_size = observed_effect_size,  # what we just calculated
                           nobs1 = n_A,         # the number of observations in A
                           alpha = 0.05,        # our alpha level
                           power = None,        # what we're interested in
                           ratio = 1            # the ratio of number of observations of A and B
                          )
power


# In[96]:


# how many observations for each of A and B would we need to get a power of .9
#   for our observed effect size and alpha level
# eg. having a 90% change of correctly deciding B is better than A
n_obs_A = tt_ind_solve_power(effect_size = observed_effect_size,  
                           nobs1 = None,         
                           alpha = 0.05,        
                           power = .9,        
                           ratio = 1            
                          )
n_obs_A


# ## Part 2: Data Cleaning and Regression

# ### Data Preparation and Exploration

# This data is provided by World Bank Open Data https://data.worldbank.org/, processed as in Homework 1.
# 
# We will be performing regression with respect to GDP and classification with respect to Income Group.
# To do that we will first need to do a little more data prep.

# In[97]:


# read in the data
df_country = pd.read_csv('../data/country_electricity_by_region.csv')

# rename columns for ease of reference
columns = ['country_code','short_name','region','income_group','access_to_electricity','gdp','population_density',
           'population_total','unemployment','region_europe','region_latin_america_and_caribbean',
           'region_middle_east_and_north_africa','region_north_america','region_south_asia',
           'region_subsaharan_africa']

df_country.columns = columns
df_country.info()


# In[98]:


# create a dummy variable 'gdp_missing' to indicate where 'gdp' is null
df_country['gdp_missing'] = df_country.gdp.isnull().astype(int)


# In[99]:


# use groupby to find the number of missing gpd by income_level
# write a lambda function to apply to the grouped data, counting the number of nulls per group
df_country.groupby('income_group').gdp.apply(lambda x: x.isnull().sum())


# In[100]:


# fill in missing gdp values according to income_group mean
# to do this, group by income_group 
# then apply a lambda function to the gdp column that uses the fillna function, filling with the mean
# inplace is not available here, so assign back into the gdp column
df_country.gdp = df_country.groupby('income_group').gdp.apply(lambda x: x.fillna(x.mean()))


# In[101]:


# assert that there are no longer any missing values in gdp
assert len(df_country[df_country.gdp.isnull()]) == 0


# In[102]:


# create 'populiation_density_missing' dummy variable
df_country['population_density_missing'] = df_country.population_density.isnull().astype(int)


# In[103]:


# fill in missing population_density with median, grouping by region
df_country.population_density = df_country.groupby('region').population_density.apply(lambda x: x.fillna(x.median()))


# In[104]:


# create a standardized 'gdp_zscore' column
from scipy.stats import zscore
df_country['gdp_zscore'] = zscore(df_country.gdp)


# In[105]:


# use seaborn to create a distplot (with rugplot indicators) and a boxplot of gdp_zscores to visualize outliers
fig, ax = plt.subplots(1,2,figsize=(12,4))
_ = sns.distplot(df_country.gdp_zscore, rug=True, ax=ax[0])
_ = sns.boxplot(df_country.gdp_zscore, ax=ax[1])


# In[106]:


# print the top 10 country_code and gdp_zscore sorted by gdp_zscore
print(df_country.sort_values(['gdp_zscore'],ascending=False).iloc[:10,:].loc[:,['country_code', 'gdp_zscore']])


# In[107]:


# set a zscore cutoff to remove the top 4 outliers
gdp_zscore_cutoff = df_country.gdp_zscore.nlargest(4).iloc[3]
gdp_zscore_cutoff


# In[108]:


# create a standardized 'population_density_zscore' column
df_country['population_density_zscore'] = zscore(df_country.population_density)


# In[109]:


# print the top 10 country_code and population_density_zscore sorted by population_density_zscore
print(df_country.sort_values(['population_density_zscore'],ascending=False).iloc[:10,:].loc[:,['country_code', 'population_density_zscore']])


# In[110]:


# set a zscore cutoff to remove the top 5 outliers
population_density_zscore_cutoff = df_country.population_density_zscore.nlargest(5).iloc[4]
population_density_zscore_cutoff


# In[111]:


# drop outliers (considering both gdp_zscore and population_density_zscore)
df_country = df_country[(df_country.gdp_zscore < gdp_zscore_cutoff) & (df_country.population_density_zscore < population_density_zscore_cutoff)]
df_country.shape


# ### Train a Regression Model

# In[112]:


# create the training set of X with features (population_density, access_to_electricity) 
# and labels y (gdp)
X = df_country[['population_density', 'access_to_electricity']]
y = df_country['gdp']


# In[113]:


# import and initialize a LinearRegression model using default parameters
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[114]:


# train the regressor on X and y
lr.fit(X,y)


# In[115]:


# print out the learned intercept and coefficients
print('intercept = {:0.3f}'.format(lr.intercept_))
print('coefficient_1 = {:0.3f}'.format(lr.coef_[0]))
print('coefficient_2 = {:0.3f}'.format(lr.coef_[1]))


# In[116]:


# we can use this mask to easily index into our dataset
country_mask = (df_country.country_code == 'CAN').values


# In[117]:


# how far off is our model's prediction for Canada's gdp (country_code CAN) from it's actual gdp?
model1_diff = abs(lr.predict(df_country.loc[country_mask, ['population_density', 'access_to_electricity']]) -     df_country.loc[country_mask, 'gdp']).values[0]
print("%e" % model1_diff)


# In[118]:


# create a new training set X that, in addition to population_density and access_to_electricity,
# also includes the region_* dummies
X = df_country[['population_density','access_to_electricity','region_europe','region_latin_america_and_caribbean',
           'region_middle_east_and_north_africa','region_north_america','region_south_asia',
           'region_subsaharan_africa']].values


# In[119]:


# instantiate a new model and train, with fit_intercept=False
lr = LinearRegression(fit_intercept=False).fit(X,y)


# In[120]:


# did the prediction for CAN improve?
model2_diff = abs(lr.predict(df_country.loc[country_mask,                              ['population_density','access_to_electricity','region_europe',                               'region_latin_america_and_caribbean','region_middle_east_and_north_africa',                               'region_north_america','region_south_asia','region_subsaharan_africa']]) -     df_country.loc[country_mask, 'gdp']).values[0]
print("%e" % model2_diff)
print('Yes') if(model1_diff > model2_diff) else print('No')

