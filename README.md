# Salary Prediction - Based on Census Data
 Predict whether income exceeds 50k per year based on census data
<h3>Description</h3>
We are using Census dataset here with almost 50000 instances and 14 features. It contains categorical and numeric type of values for differnt features. It also has missing values. To see full description of dataset please use <a href="http://archive.ics.uci.edu/ml/datasets/Adult"> Dataset Link </a> .
We are going to do all our analysis in python like observing the data, cleaning the data, transforming the data and finally training the data.
<br>
<h3>Requirements</h3>
<br>
Hope you already have some python basics and you have ready any python environment like Jupyter.
You need to install all the below libraries in your python environment. You can do this in Python IDLE using simple pip command.
<h5>numpy, pandas, scikit-learn, matplotlib</h5>
<br>
<h3> 1. Loading the Data</h3>
<br>
First we need to store the data file in your local machine somewhere and from that path we load the CSV file into Jupyter Notebook.
For reading the file, we use PANDAS library. We can also mention the delimiter type and header detais if need.
<br>

    import pandas as pd
    adult = pd.read_csv('adult.csv')

<br>
<h3> 2. Observing the Data</h3>
Now we look what are the columns in the data and what kind of values are present in the data. And we can see howmany number of rows of data is available for our analysis.
<br>

    adult.columns()
    adult.values()
    adult.shape
    adult.head()
    adult.tail()

<br>
We will see results like below.
<img src=screenshots/observing_data.jpg width=100% />
