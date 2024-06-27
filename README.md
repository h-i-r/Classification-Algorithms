# Binary Classification Algorithms

### Description
This project explores various binary classification algorithms for predictive modeling. The aim is to compare the performance of different binary classification algorithms using the famous titanic dataset. Here's the project overview:
- **Data Preparation**: Utilizes `preprocessing.py` for data cleaning and feature engineering.
- **Model Training and Evaluation**: `model.py` contains implementations of various algorithms using GridSearchCV, while `evaluate.py` assesses their performance using metrics like accuracy, precision, recall, and F1-score.
- **Visualization**: Results are visualized and stored in the `plots` directory.

Within the `model.py` we will carry out a comparative analysis on five prominent binary classification algorithms: 
- Random Forest 
- Logistic Regression
- Gradient Boosting
- Support Vector Machine
- Naïve Bayes

#### Project Structure
```
binary_classification_algorithms/
│
├── plots/                        # dir for storing plots
│
├── src/                          
│   ├── components/              
│   │   ├── evaluate.py           # evaluating the model performance
│   │   ├── model.py              # five binary classification models
│   │   └── preprocessing.py      # data preprocessing tasks
│   └── data/                     
│       └── data.csv              # dataset
│
├── main.py                       # main script to run the project
├── requirements.txt              # dependencies
└── README.md                     
```
### Usage

**1. Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**2. Run the main script:**
   ```bash
   python main.py
   ```

### Links
- [Medium Article](https://medium.com/towards-artificial-intelligence/titanic-survival-prediction-ii-551a9b44efa3)
- [Kaggle Dataset](https://www.kaggle.com/c/titanic/data)