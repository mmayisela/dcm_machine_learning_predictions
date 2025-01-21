#######Mortality Prediction using DCM healthcare data#######

#Bring in dataset

import pandas as pd

df = pd.read_csv("/Users/minenhle/Library/CloudStorage/OneDrive-UniversityofWitwatersrand/minnie_dcm_ml_projectone/heart_failure_clinical_records_dataset.csv")

df.sample(5)

print(df.sample(5))

#Separate data into X (features) and y (target)
# Separates the features into X
X = df.iloc[:,:-1].values

# Separates the target (labels) into y
y = df.iloc[:,-1].values

#Separate data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

# Standardizing the data ensures that each of the features falls within a similar range for comparison.

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)