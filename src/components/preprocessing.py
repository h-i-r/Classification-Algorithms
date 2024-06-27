import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess(df):

    df['Name'] = df['Name'].apply(lambda x: re.search(r'(Countess|Mme|Mrs|Mr|Mlle|Miss|Ms|Master|Dr|Rev|Col|Major|Capt)', x))
    df['Name'] = df['Name'].apply(lambda x: x.group() if x else 'Others')
    df['familyMembers'] = df['Parch'] + df['SibSp']
    df['ticketInfo'] = df['Ticket'].apply(lambda x: 'Type 1' if re.search(r'\d', x) is None else 'Type 2')
    df['farePerFamily'] = df['Fare'] / (df['familyMembers'] + 1)
    df['Cabin'] = df['Cabin'].str[0:1].fillna('U')

    # Impute missing values
    impute = SimpleImputer(strategy='most_frequent')
    df['Embarked'] = impute.fit_transform(df[['Embarked']])
    df['Age'] = impute.fit_transform(df[['Age']])
    df['Age'] = np.round(df['Age'], 1)
    df['ageBins'] = pd.qcut(df['Age'], q=4, labels=[0, 1, 2, 3]).astype(int)

    return df

def encode(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

def normalize(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
