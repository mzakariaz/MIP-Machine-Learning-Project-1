# Import packages
import os, sys
import numpy as np
from scipy.linalg import LinAlgWarning
import sympy as smp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wrn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_squared_log_error, root_mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_absolute_percentage_error, d2_absolute_error_score
import pickle

# Set a random seed value for code reproducibility
np.random.seed(42)

# Set a plot style for aesthetics
palette = sns.color_palette("viridis_r", as_cmap=True)
sns.set_style(style = "whitegrid", rc = {"font.family":"Times New Roman", "font.weight":"bold"})

# Ignore warnings which are not important for the purposes of this project
wrn.filterwarnings("ignore")

# Read the dataset
df = pd.read_csv("salaries.csv")

# Remove all rows with missing values
df.dropna(inplace = True)

# Convert the dataset to the right data types 
df["FIRST NAME"] = df["FIRST NAME"].astype("string")
df["LAST NAME"] = df["LAST NAME"].astype("string")
df["SEX"] = df["SEX"].astype("string")
df["DOJ"] = df["DOJ"].astype("datetime64[ns]")
df["CURRENT DATE"] = df["CURRENT DATE"].astype("datetime64[ns]")
df["DESIGNATION"] = df["DESIGNATION"].astype("string")
df["AGE"] = df["AGE"].astype("int64")
df["SALARY"] = df["SALARY"].astype("float64")
df["UNIT"] = df["UNIT"].astype("string")
df["LEAVES USED"] = df["LEAVES USED"].astype("int64")
df["LEAVES REMAINING"] = df["LEAVES REMAINING"].astype("int64")
df["RATINGS"] = df["RATINGS"].astype("float64")
df["PAST EXP"] = df["PAST EXP"].astype("int64")

# Select only the relevant features of the dataset
df_preprocessed = df[["DESIGNATION", "PAST EXP", "SALARY"]]

# One-hot encode the categorical variable
df_preprocessed = pd.get_dummies(data = df_preprocessed, columns = ["DESIGNATION"]).astype("float64").iloc[:,[2,3,4,5,6,7,0,1]]

# Scale the numerical variables
df_preprocessed[["PAST EXP", "SALARY"]] = StandardScaler().fit_transform(X = df_preprocessed[["PAST EXP", "SALARY"]])

# Split the dataset into the predictor variables and the target variable
X = df_preprocessed[["DESIGNATION_" + x for x in sorted(list(df["DESIGNATION"].unique()))] + ["PAST EXP"]].values
y = df_preprocessed["SALARY"].values

# Split the dataset into the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2, shuffle = True)

# Initialise the best model found in the salary-predictions.ipynb file, located one level above this directory
model = Lasso(alpha = 0, max_iter = 5)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make pickle file from the model
pickle.dump(model, open("pipeline/model.pkl", "wb"))