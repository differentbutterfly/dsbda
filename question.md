# DSBDA Assignments

## Group A: Data Science

### 1. Data Wrangling I

Perform the following operations using Python on any open source dataset, for example `data.csv`.

1. Import all the required Python libraries.
2. Locate an open source dataset from the web, for example from Kaggle. Provide a clear description of the data and its source, including the URL of the website.
3. Load the dataset into a Pandas DataFrame.
4. Perform data preprocessing:
   - Check for missing values in the data using `isnull()`.
   - Use the `describe()` function to get initial statistics.
   - Provide variable descriptions.
   - Mention the types of variables.
   - Check the dimensions of the DataFrame.
5. Perform data formatting and data normalization:
   - Summarize the types of variables by checking the data types, such as character, numeric, integer, factor, and logical variables.
   - If variables are not in the correct data type, apply proper type conversions.
6. Turn categorical variables into quantitative variables.

In addition to the code and outputs, explain every operation performed in the above steps and explain everything done to import, read, or scrape the dataset.

---

### 2. Data Wrangling II

Create an **Academic Performance** dataset of students and perform the following operations using Python.

1. Scan all variables for missing values and inconsistencies.
   - If there are missing values and/or inconsistencies, use suitable techniques to deal with them.
2. Scan all numeric variables for outliers.
   - If there are outliers, use suitable techniques to deal with them.
3. Apply data transformation on at least one variable.

The purpose of this transformation should be one of the following:

- To change the scale for better understanding of the variable.
- To convert a non-linear relation into a linear one.
- To decrease skewness and convert the distribution into a normal distribution.

Reason and document your approach properly.

---

### 3. Descriptive Statistics - Measures of Central Tendency and Variability

Perform the following operations on any open source dataset, for example `data.csv`.

1. Provide summary statistics such as:
   - Mean
   - Median
   - Minimum
   - Maximum
   - Standard deviation

   Perform this for a dataset, for example age, income, etc., grouped by one of the qualitative/categorical variables.

   Example:

   If the categorical variable is age group and the quantitative variable is income, then provide summary statistics of income grouped by age groups.

   Create a list that contains a numeric value for each response to the categorical variable.

2. Write a Python program to display some basic statistical details like:
   - Percentile
   - Mean
   - Standard deviation

   Perform this for the species:
   - `Iris-setosa`
   - `Iris-versicolor`
   - `Iris-virginica`

   Use the `iris.csv` dataset.

Provide the code with outputs and explain everything done in this step.

---

### 4. Data Analytics I

Create a Linear Regression model using Python/R to predict home prices using the Boston Housing dataset.

Dataset link:

`https://www.kaggle.com/c/boston-housing`

The Boston Housing dataset contains information about various houses in Boston through different parameters. There are 506 samples and 14 feature variables in this dataset.

The objective is to predict the value of house prices using the given features.

---

### 5. Data Analytics II

1. Implement Logistic Regression using Python/R to perform classification on the Social Network Ads dataset.
2. Compute the Confusion Matrix to find:
   - TP
   - FP
   - TN
   - FN
   - Accuracy
   - Error Rate
   - Precision
   - Recall

Use the given dataset.

---

### 6. Data Analytics III

1. Implement the Simple Naive Bayes classification algorithm using Python/R on the `iris.csv` dataset.
2. Compute the Confusion Matrix to find:
   - TP
   - FP
   - TN
   - FN
   - Accuracy
   - Error Rate
   - Precision
   - Recall

Use the given dataset.

---

### 7. Text Analytics

1. Extract a sample document and apply the following document preprocessing methods:
   - Tokenization
   - POS Tagging
   - Stop words removal
   - Stemming
   - Lemmatization

2. Create a representation of the document by calculating:
   - Term Frequency
   - Inverse Document Frequency

---

### 8. Data Visualization I

1. Use the inbuilt dataset `titanic`.

   The dataset contains 891 rows and contains information about the passengers who boarded the unfortunate Titanic ship.

   Use the Seaborn library to see if we can find any patterns in the data.

2. Write code to check how the price of the ticket, column name `fare`, for each passenger is distributed by plotting a histogram.

---

### 9. Data Visualization II

1. Use the inbuilt dataset `titanic`, as used in the above problem.

   Plot a boxplot for the distribution of age with respect to each gender, along with information about whether they survived or not.

   Column names:
   - `sex`
   - `age`

2. Write observations on the inference from the above statistics.

---

### 10. Data Visualization III

Download the Iris flower dataset or any other dataset into a DataFrame.

Example dataset link:

`https://archive.ics.uci.edu/ml/datasets/Iris`

Perform the following operations:

1. List down the features and their types, for example:
   - Numeric
   - Nominal

2. Create a histogram for each feature in the dataset to illustrate the feature distributions.

3. Create a boxplot for each feature in the dataset.

4. Compare distributions and identify outliers.

---

## Group B: Big Data Analytics - JAVA/SCALA

**Any three**

### 1. Word Count Application

Write code in Java for a simple Word Count application that counts the number of occurrences of each word in a given input set using the Hadoop MapReduce framework on a local standalone setup.

---

### 2. Log File Processing Using MapReduce

Design a distributed application using MapReduce that processes a log file of a system.

---

### 3. Weather Data Analysis Using MapReduce

Locate a dataset, for example `sample_weather.txt`, for working on weather data.

Read the text input files and find:

- Average temperature
- Dew point
- Wind speed

---

### 4. Scala Spark Program

Write a simple program in Scala using the Apache Spark framework.

---

## Group C: Mini Projects / Case Study - PYTHON

**Any two mini projects**

### 1. GINA Case Study

Write a case study on Global Innovation Network and Analysis, also known as GINA.

Components of analytic plan:

1. Discovery
2. Business problem framed
3. Data
4. Model planning analytic technique
5. Results and key findings

---

### 2. Tweet Sentiment Classification

Use the following dataset and classify tweets into positive and negative tweets.

Dataset link:

`https://www.kaggle.com/ruchi798/data-science-tweets`

---

### 3. Movie Recommendation Model

Develop a movie recommendation model using the scikit-learn library in Python.

Reference dataset:

`https://github.com/rashida048/Some-NLP-Projects/blob/master/movie_dataset.csv`

---

### 4. COVID-19 Vaccine State-wise Analytics

Use the following COVID vaccine state-wise dataset and perform analytics on the given dataset.

Dataset link:

`https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_vaccine_statewise.csv`

Perform the following tasks:

1. Describe the dataset.
2. Find the number of persons state-wise vaccinated for first dose in India.
3. Find the number of persons state-wise vaccinated for second dose in India.
4. Find the number of males vaccinated.
5. Find the number of females vaccinated.

---

### 5. Case Study: Digital Marketing or Healthcare Systems with Hadoop Ecosystem

Write a case study to process data-driven systems for either:

- Digital Marketing

OR

- Healthcare systems

Use Hadoop ecosystem components as shown below.

Mandatory components:

- HDFS: Hadoop Distributed File System
- YARN: Yet Another Resource Negotiator
- MapReduce: Programming-based data processing
- Spark: In-memory data processing
- Pig, Hive: Query-based processing of data services
- HBase: NoSQL database providing real-time reads and writes
- Mahout, Spark MLlib: Analytical tools and machine learning algorithm libraries
- Solr, Lucene: Searching and indexing