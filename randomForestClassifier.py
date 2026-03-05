
#___________________Random Forest Classifier___________________
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import (accuracy_score, f1_score, ConfusionMatrixDisplay, recall_score)
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

#Load dataset
df = pd.read_csv('mentalHealth.csv')

# Convert categorical strings to numeric labels
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

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

#We set up the training and testing data, w/ 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Setup Random Forest Classification
RF = RandomForestClassifier(criterion="gini", n_estimators=100, max_depth=3, random_state=42)
RF.fit(X_train, Y_train)

predY = RF.predict(X_test)

# Plot some decision trees
#Plot tree 1
fig = plt.figure(figsize=(15, 5))
tree.plot_tree(RF.estimators_[0], feature_names=X.columns, filled=True, impurity=True)
plt.show()
print(f'Accuracy on training data: {accuracy_score(Y_train, RF.predict(X_train))}')

#Plot tree 2
fig = plt.figure(figsize=(15, 5))
tree.plot_tree(RF.estimators_[1], feature_names=X.columns, filled=True, impurity=True)
plt.show()


# Predict outcome of a random record
print("Predicted risk level for record for the following: ")
print(f'Predicted risk level for record 150: {le.inverse_transform(RF.predict(X.iloc[[150]]))}')
print(f'Predicted risk level for record 50: {le.inverse_transform(RF.predict(X.iloc[[50]]))}')
print(f'Predicted risk level for record 600: {le.inverse_transform(RF.predict(X.iloc[[600]]))}')


#Confusion matrix on test data, changed df to Y_test
print(f"Unique labels in test data: {le.inverse_transform(sorted(df['gaming_addiction_risk_level'].unique()))}") #Verify labels are mapped correctly

CMatrix = ConfusionMatrixDisplay.from_predictions(Y_test, predY,
    labels=[0, 1, 2, 3], colorbar=False, display_labels=["High","Low","Moderate","Severe"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax = CMatrix.ax_
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
plt.show()

#Calculate metrics
print("Accuracy on training data: ", accuracy_score(Y_test, predY))
