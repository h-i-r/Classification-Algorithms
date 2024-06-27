import pandas as pd
from pathlib import Path
from src.components.preprocessing import (preprocess, encode, normalize)
from src.components.model import models

def main():

    current_dir = Path(__file__).resolve().parent
    file_path = current_dir/'src'/'data'/'data.csv'
    try:
        with open(file_path, 'r') as file:
            df = pd.read_csv(file)
            print(f'Data has been loaded (shape: {df.shape})')
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    df = preprocess(df)
    categorical_columns = ['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'familyMembers', 'ticketInfo', 'Cabin']
    df = encode(df, categorical_columns)
    numeric_columns = ['Fare', 'farePerFamily']
    df = normalize(df, numeric_columns)

    # Model Evaluation
    models(df)

if __name__ == "__main__":
    main()
