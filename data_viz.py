#import standard libraries
import pandas as pd
import numpy as np
import pandas_profiling
import seaborn as sns

#import user defined libraries
from constants import TRAIN_PATH,TEST_PATH,TRAIN_REPORT_PATH,TEST_REPORT_PATH

#load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

#create a profile report for the data sets
train_report = pandas_profiling.ProfileReport(train_df, title="Train Data Report")
train_report.to_file(TRAIN_REPORT_PATH)

test_report = pandas_profiling.ProfileReport(test_df, title="Test Data Report")
test_report.to_file(TEST_REPORT_PATH)

#begin visualizations on train data
#gender-wise breakdown of continuous features
var = 'customer_category'
sns.kdeplot(train_df[train_df.gender == 'Male'][var],color = 'blue')
sns.kdeplot(train_df[train_df.gender == 'Female'][var],color = 'red')

sns.histplot(train_df[train_df.gender == 'Male'][var],color = 'blue')
sns.histplot(train_df[train_df.gender == 'Female'][var], color = 'red')

#observations - negligible separation of is_active,age and vintage between male and female groups
var = 'customer_category'
sns.histplot(train_df[train_df.gender == 'Male'][var],color = 'blue')
sns.histplot(train_df[train_df.gender == 'Female'][var], color = 'red')
sns.histplot(test_df[test_df.gender == 'Male'][var],color = 'green')
sns.histplot(test_df[test_df.gender == 'Female'][var], color = 'yellow')

#multivariate plots
sns.catplot(x="customer_category", y="vintage", hue="gender", kind="violin", split=True, data=train_df)
sns.catplot(x="customer_category", y="age", hue="gender", kind="violin", split=True, data=train_df)
sns.catplot(x="city_category", y="vintage", hue="gender", kind="violin", split=True, data=train_df)
sns.catplot(x="city_category", y="age", hue="gender", kind="violin", split=True, data=train_df)
sns.catplot(x="is_active", y="vintage", hue="gender", kind="violin", split=True, data=train_df)
sns.catplot(x="is_active", y="age", hue="gender", kind="violin", split=True, data=train_df)

# Observations - is_active, customer_category and city_category do not reveal any meaningful differences
# in the distribution of age and tenure / vintage. 
# Conclusions - we may want to exclude these from the model building process