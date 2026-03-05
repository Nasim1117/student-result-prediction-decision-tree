import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Create the dataset

student_data = {
    "StudyHours": [2, 4, 5, 6, 1, 7, 3, 8, 2, 6],
    "Attendance": [60, 70, 75, 82, 50, 88, 68, 92, 55, 80],
    "PreviousMarks": [48, 60, 66, 72, 40, 85, 58, 90, 45, 74],
    "Result": ["Fail", "Pass", "Pass", "Pass", "Fail", "Pass", "Fail", "Pass", "Fail", "Pass"]
}

dataset = pd.DataFrame(student_data)

print("Student Dataset:")
print(dataset)

features = dataset[["StudyHours", "Attendance", "PreviousMarks"]]

target = dataset["Result"]

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=1
)

# Create Decision Tree model

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Predict new student result

new_student = [[4, 72, 65]]
prediction = dt_model.predict(new_student)
print("\nPrediction for new student:", prediction[0])

# Check model accuracy

accuracy = dt_model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Visualize Decision Tree

plt.figure(figsize=(14, 8))

tree.plot_tree(
    dt_model,
    feature_names=features.columns,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - Student Result Prediction")

plt.show()