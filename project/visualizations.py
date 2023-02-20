import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the training data
frame = pd.read_csv("./data/train.csv")

# Cleansing some data
frame['Age'] = frame['Age'].fillna(frame['Age'].mean())
frame = frame.drop(columns="Cabin", axis=1)

frame['Succumb'] = 1 - frame['Survived']

# Visualization into the data
frame['Succumb'] = 1 - frame['Survived']

# Based on sex
sex_discriminator = frame.groupby('Sex').agg('mean')[['Survived', 'Succumb']]

bar_x = ['male', 'female']
bar_y1 = np.array([sex_discriminator['Survived']['male'],
                   sex_discriminator['Survived']['female']])
bar_y2 = np.array([sex_discriminator['Succumb']['male'],
                   sex_discriminator['Succumb']['female']])

plt.bar(bar_x, bar_y1, color="green")
plt.bar(bar_x, bar_y2, bottom=bar_y1, color='red')

plt.xlabel = "Gender"
plt.ylabel = "Number of Passengers"
plt.legend(["Survived", "Succumb"])
plt.title("ML Project - Ratio of survived:dead people based on their sex")

plt.show()

sns.violinplot(x='Sex', y='Age',
               hue='Survived', data=frame,
               split=True,
               palette={0: "r", 1: "g"}
               )
plt.title("ML Project - Age and Sex plot")
plt.show()

# Based on Fare
plt.hist([frame[frame['Survived'] == 1]['Fare'], frame[frame['Survived'] == 0]['Fare']],
         stacked=True, color=['g', 'r'],
         bins=50, label=['Survived', 'Succumb'])

plt.xlabel = "Fare"
plt.ylabel = "Number of Passengers"
plt.legend()
plt.title("ML Project - Plot of survived:dead people based on their fare class")

plt.show()

print(frame.Fare.describe())
frame['Fare_Category'] = pd.cut(frame['Fare'], bins=[0, 7.90, 14.45, 31.28, 120], labels=['<25%', '25-50%',
                                                                                          '50-75%', '>75%'])

x = sns.countplot(x="Fare_Category", hue="Survived", data=frame, palette=[
                  "C1", "C0"]).legend(labels=["Succumbed", "Survived"])
x.set_title("Survival based on fare category")

plt.show()

# Based on siblings
print(frame.SibSp.describe())
ss = pd.DataFrame()
ss['survived'] = frame.Survived
ss['sibling_spouse'] = pd.cut(
    frame.SibSp, [0, 1, 2, 3, 4, 5, 6, 7, 8], include_lowest=True)
(ss.sibling_spouse.value_counts()).plot.area().set_title(
    "Number of siblings or spouses:survival chance")

plt.show()

x = sns.countplot(x="sibling_spouse", hue="survived", data=ss, palette=[
                  "C1", "C0"]).legend(labels=["Succumbed", "Survived"])
x.set_title("Survival based on number of siblings or spouses")

plt.show()

# Based on parch - number of parents orr children
print(frame.Parch.describe())
ss = pd.DataFrame()
ss['survived'] = frame.Survived
ss['parents_children'] = pd.cut(
    frame.Parch, [0, 1, 2, 3, 4, 5, 6, 7, 8], include_lowest=True)
(ss.parents_children.value_counts()).plot.area().set_title(
    "Number of parents or children:survival chance")

plt.show()

x = sns.countplot(x="parents_children", hue="survived", data=ss, palette=[
                  "C1", "C0"]).legend(labels=["Succumbed", "Survived"])
x.set_title("Survival based on number of parents or children")

plt.show()
