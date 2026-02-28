import pandas as pd

df = pd.read_csv('Add file here')

def extractData(column):
    print(f"{column} - Mean: {df[column].mean()}, Min: {df[column].min()}, Max: {df[column].max()} \n")


#Data on grades before filling in missing data
extractData('grades_gpa')
print(f"Number of missing values in 'grades_gpa': {df['grades_gpa'].isnull().sum()}")


#Fills in missing data for grades_gpa with median of the column
median_gpa = df['grades_gpa'].median()
print(f"Median GPA: {median_gpa}")
df.loc[(df['grades_gpa'].isnull()), 'grades_gpa'] = median_gpa


#Data on grades after filling in missing data
extractData('grades_gpa')
print(f"Number of missing values in 'grades_gpa': {df['grades_gpa'].isnull().sum()}")

#___________________Data Exploration___________________
#prints first 5 rows
print(df[:5])

#Prints number of empty spaces in each column
#Empty spaces in both grades_gpa and work_productivity_score
#Could be that not everyone who works is currently attending school and vice versa 
for column in df.columns:
    
    if df[column].isnull().sum() > 0:
        print(f"Number of empty spaces in '{column}' column: {df[column].isnull().sum()}")
    else:
        print(f"No empty spaces in '{column}' column")
    
#Data on age
extractData('age')

#data on gender
for gender in df['gender'].unique():
    print(f"Number of people who are {gender}: {(df['gender'] == gender).sum()}")

#Data on gaming hours
extractData('daily_gaming_hours')

#Data on sleep hours
extractData('sleep_hours')

#data on genre
for genre in df['game_genre'].unique():
    print(f"Number of people who play {genre}: {(df['game_genre'] == genre).sum()}")

#Exporting cleaned data to a new csv file
#df.to_csv('cleaned_mentalHealth.csv')