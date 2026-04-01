from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, recall_score)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def askFloat(prompt, minVal = None, maxVal = None):
    try:
         val = float(input(prompt))
         if (minVal is not None and val < minVal) or (maxVal is not None and val > maxVal):
             print(f"Please enter a value between {minVal} and {maxVal}.")
             return askFloat(prompt, minVal, maxVal)
         else:
             return val
    except ValueError:
        print("Please enter a valid number.")
        return askFloat(prompt, minVal, maxVal)

def askYesNo(prompt):

    val = input(prompt)

    if val.lower() in ['yes', 'y', 'true', '1']:
        return True
    elif val.lower() in ['no', 'n', 'false', '0']:
        return False
    else:
        print("Please enter a valid response (yes/no).")
        return askYesNo(prompt)

def askSurvey():
    surveyData = {

        "daily_gaming_hours": askFloat("In the last 30 days, on average, how many hours per day do you spend playing video games?\n", 0, 24),
        "loss_of_other_interests": askYesNo("Have you lost interest in hobbies or activities that you used to enjoy?"),
        "withdrawal_symptoms": askYesNo("Have you experienced symptoms of withdrawal when you are not able to play games?"),
        "back_neck_pain": askYesNo("Have you experienced back or neck pain related to gaming?"),
        "face_to_face_social_hours_weekly": askFloat("On average, how many hours per week do you spend on face-to-face social interactions?\n", 0, 168),
        "monthly_game_spending_usd": askFloat("On average, how much money do you spend on video games per month (can include in-game purchases)?\n", 0),
        "social_isolation_score": askFloat("How socially isolated have you felt? (1 = not at all, 10 = extremely isolated)?\n", 1, 10),
        "continued_despite_problems":  askYesNo("Have you continued to play games even when it caused problems?\n"),
        "sleep_hours": askFloat("On average, how many hours do you sleep per night?\n", 0, 24),
        "exercise_hours_weekly": askFloat("On average, how many hours per week do you spend exercising?\n", 0, 168)
    }

    surveyDF = pd.DataFrame([surveyData])
    
    return surveyDF

def predictionModel(trainingDF, surveyDF, previousUserDF = None):
    # Convert categorical strings to numeric labels
    le = LabelEncoder()
    for column in trainingDF.columns:
        if trainingDF[column].dtype == 'object':
            trainingDF[column] = le.fit_transform(trainingDF[column])

    # Define features and target variable
    X = trainingDF[[
        "daily_gaming_hours",
        "loss_of_other_interests",
        "withdrawal_symptoms",
        "back_neck_pain",
        "face_to_face_social_hours_weekly",
        "monthly_game_spending_usd",
        "social_isolation_score",
        "continued_despite_problems",
        "sleep_hours",
        "exercise_hours_weekly"
    ]]
    
    y = trainingDF['gaming_addiction_risk_level']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Classifier
    RF = RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=3, random_state=42)
    RF.fit(X_train, y_train)
    # Predict on the test set
    y_pred = RF.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Recall Score:", recall_score(y_test, y_pred, average='weighted'))

    # Predict on the survey data
    surveyPrediction = RF.predict(surveyDF)
    surveyDF['gaming_addiction_risk_level'] = le.inverse_transform(surveyPrediction)
    print(f"Based on your responses, your predicted gaming addiction risk level is: {le.inverse_transform(surveyPrediction)[0]}")

    # load previous survey results if they exist and add the new responses and prediction to the previous results, then export to a new csv file
    if previousUserDF is not None:
        updatedUserDF = pd.concat([previousUserDF, surveyDF], ignore_index=True)
        updatedUserDF.to_csv(f'{input("Please input the file path where you want to save the updated results:\n").strip(" \"\'")}.csv', index=False)
        print("Your new survey results have been added to your previous results and exported to the specified file.")

    else:
        surveyDF.to_csv(f'{input("Please input the file path where you want to save your survey results:\n").strip(" \"\'")}.csv', index=False)
        print("Your survey results have been exported to the specified file.")


trainingDF = pd.read_csv('cleaned_mentalHealth.csv')

print("Welcome to the Gaming Addiction Risk Level Survey! Please answer the following questions to the best of your ability. For yes and no questions, please respond with 'yes' or 'no'.\n")
print("Please note that this survey is not a diagnostic tool and is only intended to provide an estimate of your gaming addiction risk level based on the data provided.\n")

previousUser = input("Before we begin, have you taken this survey previously and downloaded a csv of your results? (yes/no)\n")

if previousUser.lower() in ['yes', 'y']:
    previousUserDF = pd.read_csv(input("Please input file path of your previous survey results CSV:\n").strip(' "\''))
else:    previousUserDF = None

print("\nThank you for providing that information. Let's begin the survey\n")

surveyDF = askSurvey()

predictionModel(trainingDF, surveyDF, previousUserDF)



"""
daily_gaming_hours                  0.360845
loss_of_other_interests             0.136124
withdrawal_symptoms                 0.132598
back_neck_pain                      0.080239
face_to_face_social_hours_weekly    0.060720
monthly_game_spending_usd           0.060384
social_isolation_score              0.049419
continued_despite_problems          0.043612
sleep_hours                         0.036417
exercise_hours_weekly               0.014173

possibly use another game to help w/ addiction?

In the last 30 days, on average, how many hours per day do you spend playing video games?
Have you lost interest in hobbies or activities that you used to enjoy?
Have you experienced symptoms of withdrawal when you are not able to play games?
Have you experienced back and/or neck pain related to gaming?
How many hours per week did you spend in face-to-face social activities
How much money do you spend monthly on games (including in game purchases)?
How socially isolated have you felt? (1 = not at all, 10 = extremely isolated)
Have you continued to play games even when it caused problems?
On average, how many hours do you sleep per night?
On average, how many hours do you exercise per week?
"""