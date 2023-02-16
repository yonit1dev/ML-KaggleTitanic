import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as plot

# Function that will be useful later


def calculate_median(median_frame, observation):
    selection = (
        (median_frame['Status'] == observation['Status']) &
        (median_frame['Sex'] == observation['Sex']))
    return median_frame[selection]['Age'].values[0]


# Reading the training data
frame = pd.read_csv("./data/train.csv")

# Check for the right amount of columns and rows
print(frame.shape == (891, 12))

# Manipulating data to be accomodated by model
# Combining train and test data


def combine_train_test(train_frame, test):
    test_data = pd.read_csv(test)

    # class_target = train_frame.Survived
    train_frame.drop(columns="Survived", axis=1, inplace=True)

    combined_data = train_frame.append(test_data)
    combined_data.reset_index(inplace=True)
    combined_data.drop(
        columns=['PassengerId', 'index', 'Cabin'], axis=1, inplace=True)

    return combined_data


data_set = combine_train_test(frame, "./data/test.csv")
# print(data_set.shape)
# print(data_set.head())

# Manipulating each of the features to create a model-friendly data
# Age has some values set as null

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

# print(data_set.head())

# Processing Age
# Filling null values with their median values by sex and status category
train_group = data_set.iloc[:891].groupby(['Status', 'Sex'])
train_median = train_group.median().reset_index()[['Sex', 'Status', 'Age']]

# print(train_median.head())

# based on the above observation and computation, fill the age column with the median value if null
data_set["Age"] = data_set.apply(lambda row: calculate_median(median_frame=train_median,
                                                              observation=row) if np.isnan(row["Age"]) else row["Age"], axis=1)

# Processing Sex
data_set["Sex"] = data_set["Sex"].map({'male': 0, 'female': 1})

# Dropping the name column
data_set.drop(columns="Name", axis=1, inplace=True)
print(data_set.shape)

# Replacing embarked columns with the mode
data_set.Embarked.fillna('S', inplace=True)

# Changing values of embarked to numbers; 0 for S, 1 for C, 2 for Q
data_set["Embarked"] = data_set["Embarked"].map({'S': 0, 'C': 1, 'Q': '2'})
# print(data_set.head())

# Checking for null values in train set
print(data_set.iloc[:891].isnull().sum())

