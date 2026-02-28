import pandas as pd

df = pd.read_csv(r'add file path here')

#prints first 5 rows
print(df[:5])

#Prints number of empty spaces in each column
#Empty spaces in both grades_gpa and work_productivity_score
#Could be that not everyone who works is currently attending school and vice versa 
for column in df.columns:
    
    if df[column].isnull().sum() > 0:
        print(f"Number of empty spaces in '{column}' column: \n {(df[column].isnull()).sum()}")
    else:
        print(f"No empty spaces in '{column}' column")
    
#Data on age
print(f"{'age'} - Mean: {df['age'].mean()}, Min: {df['age'].min()}, Max: {df['age'].max()}")

#data on gender
for gender in df['gender'].unique():
    print(f"Number of people who are {gender}: {(df['gender'] == gender).sum()}")

#There is 1000 people in the study, does not take into account those studying and working or those doing neither
print(f"of these people, {df['grades_gpa'].notnull().sum()} are attending school and {df['work_productivity_score'].notnull().sum()} are working")

#Data on gaming hours
print(f"{'daily_gaming_hours'} - Mean: {df['daily_gaming_hours'].mean()}, Min: {df['daily_gaming_hours'].min()}, Max: {df['daily_gaming_hours'].max()}")

#Data on sleep hours
print(f"{'sleep_hours'} - Mean: {df['sleep_hours'].mean()}, Min: {df['sleep_hours'].min()}, Max: {df['sleep_hours'].max()}")

#data on genre
for genre in df['game_genre'].unique():
    print(f"Number of people who play {genre}: {(df['game_genre'] == genre).sum()}")