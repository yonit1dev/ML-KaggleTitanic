import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function that will be useful later
def calculate_median(median_frame, observation):
    selection = (
        (median_frame['Status'] == observation['Status']) &
        (median_frame['Sex'] == observation['Sex']))
    return median_frame[selection]['Age'].values[0]


def combine_train_test(train_frame, test):
    test_data = pd.read_csv(test)

    train_frame.drop(columns="Survived", axis=1, inplace=True)

    combined_data = pd.concat([train_frame, test_data])
    combined_data.reset_index(inplace=True)
    combined_data.drop(
        columns=['PassengerId', 'index', 'Cabin'], axis=1, inplace=True)

    return combined_data


# Reading the training data
frame = pd.read_csv("./data/train.csv")

# Check for the right amount of columns and rows
print(frame.shape == (891, 12))

# Manipulating data to be accomodated by model
# Combining train and test data
data_set = combine_train_test(frame, "./data/test.csv")

# Manipulating each of the features to create a model-friendly data

# Extracting useful information from the title columns - indicates social status
titles = {
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Dr": "Officer",
    "Rev": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "the Countess": "Royalty",
    "Master": "Master",
    "Lady": "Royalty"
}

data_set["Status"] = data_set["Name"].map(
    lambda name: name.split(",")[1].split(".")[0].strip())
data_set["Status"] = data_set.Status.map(titles)

# Processing Age
# Filling null values with their median values by sex and status category
train_group = data_set.iloc[:891].groupby(['Status', 'Sex'])
train_median = train_group.median(numeric_only=True).reset_index()[['Sex', 'Status', 'Age']]

# based on the above observation and computation, fill the age column with the median value if null
data_set["Age"] = data_set.apply(lambda row: calculate_median(median_frame=train_median,
                                                              observation=row) if np.isnan(row["Age"]) else row["Age"], axis=1)

# Processing Sex
data_set["Sex"] = data_set["Sex"].map({'male': 0, 'female': 1})

# Dropping the name column
data_set.drop(columns="Name", axis=1, inplace=True)
print(data_set.shape)

# Replacing null embarked columns with the mode
data_set.Embarked.fillna(data_set["Embarked"].mode()[0], inplace=True)

# Changing values of embarked to numbers; 0 for S, 1 for C, 2 for Q
data_set["Embarked"] = data_set["Embarked"].map({'S': 0, 'C': 1, 'Q': '2'})

# Replacing null fare columns with mean
data_set.Fare.fillna(data_set["Fare"].mean(), inplace=True)

# Converting status to numerical values
status_dummies = pd.get_dummies(data_set["Status"], prefix="Status")
data_set = pd.concat([data_set, status_dummies], axis=1)
data_set.drop(columns="Status", axis=1, inplace=True)

# Dropping other irrelevant columns
data_set.drop(columns=["SibSp", "Parch", "Ticket"], axis=1, inplace=True)


#Modeling the data
target_class = pd.read_csv('./data/train.csv',
                           usecols=['Survived'])['Survived'].values
train_set = data_set.iloc[:891]
test_set = data_set.iloc[891:]

# Selecting the test set to be 20%
x_train = train_set[:713]
x_test = train_set[713:]
target_train = target_class[:713]
target_test = target_class[713:]

logisitic = LogisticRegression(max_iter=500).fit(x_train, target_train)

# training data prediction
prediction = logisitic.predict(x_train)

train_accuracy = accuracy_score(target_train, prediction)
print('Accuracy of training data : ', train_accuracy)

test_prediction = logisitic.predict(x_test)
test_accuracy = accuracy_score(target_test, test_prediction)
print('Accuracy of test data : ', test_accuracy)

# predict test data
test_predict = logisitic.predict(test_set)
print(test_predict)
