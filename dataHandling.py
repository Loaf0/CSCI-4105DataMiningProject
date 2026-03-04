import pandas as pd

df = pd.read_csv('mentalHealth.csv')

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

#Examine possible value of gaming addiction risk level column
print(df["gaming_addiction_risk_level"].unique())

#___________________Random Forest Classifier___________________
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import (accuracy_score, f1_score, ConfusionMatrixDisplay, recall_score)

from sklearn.ensemble import RandomForestClassifier

# Convert categorical strings to numeric labels
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

print(df.head())

# Create dependent and independent variable
Y=df["gaming_addiction_risk_level"]

X = df[[
    "age",
    "gender",
    "daily_gaming_hours",
    "game_genre",
    "primary_game",
    "gaming_platform",
    "sleep_hours",
    "sleep_quality",
    "sleep_disruption_frequency",
    "academic_work_performance",
    "grades_gpa",
    "work_productivity_score",
    "mood_state",
    "mood_swing_frequency",
    "withdrawal_symptoms",
    "loss_of_other_interests",
    "continued_despite_problems",
    "eye_strain",
    "back_neck_pain",
    "weight_change_kg",
    "exercise_hours_weekly",
    "social_isolation_score",
    "face_to_face_social_hours_weekly",
    "monthly_game_spending_usd",
    "years_gaming",
]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Setup Random Forest Classification
"""
n_estimators: how many decision tree
min_sample_split: minimum data points required to split a node
"""
RF = RandomForestClassifier(criterion="gini", n_estimators=100, max_depth=3, random_state=42)

# Apply to training data
RF.fit(X_train, Y_train)

# (Optional) label decision tree

# Plot some decision trees
#Plot tree 1
fig = plt.figure(figsize=(15, 5))
tree.plot_tree(RF.estimators_[0], filled=True, impurity=True)
plt.show()

#Plot tree 2
fig = plt.figure(figsize=(15, 5))
tree.plot_tree(RF.estimators_[1], filled=True, impurity=True)
plt.show()

# Predict outcome of a random record
print(RF.predict(X.iloc[[150]]))
print(RF.predict(X.iloc[[50]]))
print(RF.predict(X.iloc[[600]]))

# Apply the classifier to test data
Y_test = pd.DataFrame(Y_test)
Y_test["Predicted"] = RF.predict(X_test)
print(Y_test.head())

#Confusion matrix on test data
print(sorted(df["gaming_addiction_risk_level"].unique())) #Verify labels are mapped correctly

ConfMatrix = ConfusionMatrixDisplay.from_predictions(Y_test["gaming_addiction_risk_level"], Y_test["Predicted"],
    labels=[0, 1, 2, 3], colorbar=False, display_labels=["High","Low","Moderate","Severe"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax = ConfMatrix.ax_
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
plt.show()

#Calculate metrics
print("Accuracy: ", metrics.accuracy_score(Y_test["gaming_addiction_risk_level"], Y_test["Predicted"]))