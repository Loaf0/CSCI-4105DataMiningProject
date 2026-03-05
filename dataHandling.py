import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

#Read and drop the csv file, dropping the record_id column since it is not needed for analysis
df = pd.read_csv(input("Please input file path of CSV:\n").strip(' "\''))
#df.drop(columns=['record_id'], inplace=True)

#Extracts basic statistics from numeric columns in the dataset
def extractNumericData(column):
    print(f"{column} - Mean: {df[column].mean()}, Min: {df[column].min()}, Max: {df[column].max()} \n")

def extractStringData(column):
    unique_values = df[column].dropna().unique().tolist()
    print(f"{column} - Unique Values: {unique_values}")

    for value in unique_values:
        count = (df[column] == value).sum()
        print(f"Number of people with {value} in {column} column: {count}")
    print()

#___________________Find Missing Data___________________
#Looks through each column of the dataset and prints out the number of empty spaces in each column
#except for the record_id column
for column in df.columns:
    
    if df[column].isnull().sum() > 0:
        print(f"Number of empty spaces in '{column}' column: {df[column].isnull().sum()} \n")
    else:
        print(f"No empty spaces in '{column}' column \n")


#___________________Fill Missing Data___________________
#Fills in missing data for grades_gpa with median of the column
median_gpa = df['grades_gpa'].median()
print(f"Median GPA: {median_gpa}")
df.loc[(df['grades_gpa'].isnull()), 'grades_gpa'] = median_gpa

#___________________Data Exploration___________________
#prints first 5 rows
print(df[:5])

#Extracts basic statistics from int columns in the dataset
for column in df.columns:
    if is_numeric_dtype(df[column]):
        extractNumericData(column)
    else:
        extractStringData(column)


#Exporting cleaned data to a new csv file
df.to_csv('cleaned_mentalHealth.csv')

#Examine possible value of gaming addiction risk level column
print(df["gaming_addiction_risk_level"].unique())