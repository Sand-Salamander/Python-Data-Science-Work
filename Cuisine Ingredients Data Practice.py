# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 08:07:50 2020

@author: sande
"""
import pandas as pd
import numpy as np
import re

%matplotlib inline
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

#!conda install python-graphviz --yes
import graphviz

from sklearn.tree import export_graphviz

import itertools

#Loading the data from the API
recipes = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DS0103EN/labs/data/recipes.csv")

#Gaining a better understanding of the data
recipes.head()
recipes['country'].value_counts()

#Changing the 'country' column to properly display 'cuisine' lable 
recipes.columns.values[0] = 'cuisine'
recipes

#Convert the Yes/No responses to boolean values 0/1
recipes = recipes.replace(to_replace="Yes", value=1)
recipes = recipes.replace(to_replace="No", value=0)
recipes

#Check all recipes that contain butter and rice
check = recipes.loc[(recipes['rice'] ==1) & (recipes['butter']==1)]
check
check['butter']
#this last line will show the row indexes that contain butter. It should match with the row indexes identified by the check variable

#Count the ingredients by first summing up the columns
total_count = recipes.iloc[:,1:].sum(axis=0)
total_count

#Create a dataframe that includes the total_count of each ingredient in descending order
ingredient = pd.Series(total_count.index.values, index = np.arange(len(total_count)))
count = pd.Series(list(total_count), index = np.arange(len(total_count)))
total = pd.DataFrame(dict(ingredient = ingredient, count = count))
total = total[['ingredient', 'count']]
total.sort_values(['count'], ascending = False, inplace = True)
total.reset_index(inplace=True, drop = True)
print(total.to_string())



#Can now analyze these results to determine which ingredients are most popular, on average, in each cuisine
cuisines = recipes.groupby(recipes.columns[0]).mean()
cuisines.head()

#Create a function that creates a profile for each cuisine's top 3 ingredients 
num_ingredients = 3
def print_top_ingredients(row):
    print(row.name.upper())
    row_sorted = row.sort_values(ascending=False)*100
    top_ingredients = list(row_sorted.index.values)[0:num_ingredients]
    row_sorted = list(row_sorted)[0:num_ingredients]

    for ind, ingredient in enumerate(top_ingredients):
        print("%s (%d%%)" % (ingredient, row_sorted[ind]), end=' ')
    print("\n")

create_cuisines_profiles = cuisines.apply(print_top_ingredients, axis=1)

#Creating a Decision Tree from A Subset of Asian Recipes
asian_indian_recipes = recipes[recipes.cuisine.isin(["korean", "japanese", "chinese", "thai", "indian"])]
cuisines = asian_indian_recipes["cuisine"]
ingredients = asian_indian_recipes.iloc[:,1:]
bamboo_tree = tree.DecisionTreeClassifier(max_depth=3)
bamboo_tree.fit(ingredients, cuisines)

export_graphviz(bamboo_tree,
                feature_names=list(ingredients.columns.values),
                out_file="bamboo_tree.dot",
                class_names=np.unique(cuisines),
                filled=True,
                node_ids=True,
                special_characters=True,
                impurity=False,
                label="all",
                leaves_parallel=False)

with open("bamboo_tree.dot") as bamboo_tree_image:
    bamboo_tree_graph = bamboo_tree_image.read()
graphviz.Source(bamboo_tree_graph)

