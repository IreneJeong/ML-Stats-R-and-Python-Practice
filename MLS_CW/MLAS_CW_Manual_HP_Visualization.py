#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:11:59 2023

@author: jeongdahye
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset into a pandas dataframe
df = pd.read_csv("mydataset.csv")

df['OPTIMIZER'] = df['OPTIMIZER'].astype('category').cat.codes
df['Model'] = df['Model'].astype('category').cat.codes

g = sns.pairplot(df, hue='test_Accuracy')
g.map_upper(sns.scatterplot).set(ylim=(0, None))
