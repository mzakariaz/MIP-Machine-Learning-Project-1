<h1><b>MIP-Machine-Learning-Project-1</b></h1>

<h2><b>Contents</b></h2>

- [**Introduction**](#introduction)
- [**Details**](#details)
- [**Credits**](#credits)

## **Introduction**
This project is my attempt at Project 1 of the Machine Learning Engineer internship at Mentorness. The goal is to build an accurate and successful predictive model on the salaries of data professionals.

## **Details**
* **<u>Project Title:</u>** *Salary Predictions of Data Professionals*
* **<u>Project Description:</u>** Salaries in the field of data professions vary widely based on factors such as experience, job role, and performance. Accurately predicting salaries for data professionals is essential for both job seekers and employers.  
* **<u>Project Submission:</u>** All of the work to be evaluated for this project is found in the `salary-predictions.ipynb` file, located in the current directory, and in the `pipeline` folder, also located in the current directory.
* **<u>Libraries Used:</u>** The following libraries were used for the stated purposes:
  - `os`, for file handling;
  - `sys`, for system handling;
  - `numpy`, for numerical calculations;
  - `scipy.linalg`, to import the `LinAlgWarning` warning and suppress it;
  - `sympy`, for symbolic mathematics;
  - `pandas`, for data analysis;
  - `matplotlib.pyplot`, for data visualisation;
  - `seaborn`, for data visualisation;
  - `warnings`, for warning handling;
  - `sklearn.preprocessing`, for preprocessing datasets for machine learning model implementation, especially through the `StandardScaler` module;
  - `sklearn.model_selection`, for making selections regarding the dataset, especially through the `train_test_split`, `cross_val_score` and `GridSearchCV` modules;
  - `sklearn.linear_model`, for implementing linear regression models, especially through the `LinearRegression`, `Ridge` and `Lasso` modules;
  - `sklearn.metrics`, for evaluating the performance of machine learning models, especially through the `explained_variance_score`, `max_error`, `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `mean_squared_log_error`, `root_mean_squared_log_error`, `median_absolute_error`, `r2_score`, `mean_poisson_deviance`, `mean_gamma_deviance`, `mean_absolute_percentage_error` and `d2_absolute_error_score` modules;
  - `pickle`, for loading and running machine learning pipelines on web applications, and;
  - `flask`, for creating web applications and running them on a server, especially through the `Flask`, `request`, `jsonify` and `render_template` modules.

## **Credits**
All of the sources (e.g. videos, images, articles) used to help build this project are cited in the links below:
- https://www.explorium.ai/blog/machine-learning/feature-engineering/#:~:text=One%20example%20of%20feature%20engineering,of%20this%20kind%20of%20data.
- https://www.statology.org/seaborn-barplot-order/
- https://towardsdatascience.com/hyperparameter-tuning-in-lasso-and-ridge-regressions-70a4b158ae6d
- https://en.wikipedia.org/wiki/Ridge_regression
- https://en.wikipedia.org/wiki/Lasso_(statistics)
- https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/#:~:text=Lasso%20regression%20is%20preferred%20when,interpretable%20model%20with%20fewer%20variables.&text=Ridge%20regression%20tends%20to%20favor,keeps%20them%20in%20the%20model.
- https://www.youtube.com/watch?v=GN6ICac3OXY
- https://www.youtube.com/watch?v=-ykeT6kk4bk
- https://www.youtube.com/watch?v=d6Jw5hGb65Y
- https://www.youtube.com/watch?v=MxJnR1DMmsY