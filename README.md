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
<br>
<img src=screenshots/observing_data.jpg width=70% />
<br>

<h3> 3. Cleaning the Data</h3>
<h5> 3.1 Remove Unwanted Rows </h5>
From the above analysis, we see fields 'education' and 'education_num' are two fields which are interrelated and we can use only field for analysis. Because this might lead to multi-colleniarity problem. And field 'finalwgt' is another informative field, we can remove this too.
<br>
So this is our next step, we are gonna drop those columns. This can be done in many ways, we can explicitly mention all the fields we gonna keep or drop those 2 fields.

    mydata = adult.iloc[:,[0,1,4,5,6,7,8,9,10,11,12,13,14]]
    mydata
    #adult.drop(['fnlwgt', 'education'])

<h5> 3.2 Look for null values </h5>
Now we have to check for any null values in the dataset. Null values are 'na' values which can be any of the default null values present in PANDAS library like Nan, #NA,.. etc.

     mydata.isnull().any()
     mydata.isnull().sum()
     mydata[pd.isnull(mydata['Age'])] # to check only in Age column.

We find some values like 'No Gender' in Age column, so we replace these values with default 'na' values.

     mydata["Age"].fillna("No Gender", inplace = True)

<h5> 3.3 Removing '?' Rows </h5>
There are also some values with '?' in the dataset. These can also be replaced with 'na' values like above. Or while reading the file itself we can mention null values with an attribute. So here first we find out what are the columns have '?' values.

     for each_column in mydata.columns:
         if ' ?' in mydata[each_column].values:
             print (each_column)

This gives 3 columns as 'workclass', 'occupation' and 'native-country'.
<br>
We also want to see howmany such '?' rows are existing in each column. This is the step by step processing of missing values. This is the most important step in any Pre-processing.

     mydata.loc[mydata['workclass']==' ?'].shape  #1836
     mydata.loc[mydata['occupation']==' ?'].shape  #1843
     mydata.loc[mydata['native-country']==' ?'].shape  #583

Now what do we do with these rows, these are not numerical values to replace with mean or median. These are important categorical values. If we see the data size, from total around 50K rows if we remove like 1800 rows , that would be a best approach as we won't mislead data with inappropriate data. So let's remove them.

     final_data=mydata[mydata.workclass!=' ?']
     final_data=final_data[final_data.occupation!=' ?']

<h5> 3.4 Transform data from charcter to numeric </h5>
Now we have to do some transformations to the character variables to numeric values. We do this to apply an ML algorithm. For an algorithm its all numbers, for eg, if we have values as 'private' and 'public' it converts them to 0 and 1 or may be 1 and 2. This way we can apply a classification prediction on the dataset.
<br>
In Python we have 'LabelEncoder' library to do that for us. So we transform each column one by one.

     final_data['workclass']=LabelEncoder().fit_transform(final_data['workclass'])
     final_data['marital_status']=LabelEncoder().fit_transform(final_data['marital_status'])
     final_data['occupation']=LabelEncoder().fit_transform(final_data['occupation'])
     final_data['relationship']=LabelEncoder().fit_transform(final_data['relationship'])
     final_data['race']=LabelEncoder().fit_transform(final_data['race'])
     final_data['sex']=LabelEncoder().fit_transform(final_data['sex'])
     final_data['native-country']=LabelEncoder().fit_transform(final_data['native-country'])
     
<h3> 4. Splitting Train and Test data</h3>
