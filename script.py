
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import scipy.io
import sys
import subprocess
from platform import python_version
from random import sample
from random import seed
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import seaborn as sns
from math import ceil

%matplotlib inline

#%%



# Some plot styling preferences
plt.style.use('seaborn-whitegrid')
font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 14}

mpl.rc('font', **font)
effect_size = sms.proportion_effectsize(0.13, 0.15)    # Calculating effect size based on our expected rates

required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
    )                                                  # Calculating sample size needed

required_n = ceil(required_n)                          # Rounding up to next whole number                          

print(required_n)


#%%
df_bridebook = pd.read_csv('/Users/ahmedadel/Downloads/Bridebook - Data Scientist , Task/Bridebook - DS - challengeData.1634399212.csv')
df_description = pd.read_excel('/Users/ahmedadel/Downloads/Bridebook - Data Scientist , Task/Bridebook - DS - challengeDataLegend.1634399212.xlsx')