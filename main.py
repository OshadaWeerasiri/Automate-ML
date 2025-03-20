from autogluon.tabular import TabularDataset,  TabularPredictor
from sklearn.model_selection import train_test_split
import pandas as pd

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = TabularDataset(url)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")


# Define target variable (what we want to predict)
target = 'Survived'

# Train the model
predictor = TabularPredictor(label=target).fit(
    train_data=train_data,
    time_limit=120,  # 2 minutes for quick results (increase for better accuracy)
    presets='best_quality'  # Options: 'medium_quality', 'high_quality' (faster vs. slower)
)

# Generate predictions
y_pred = predictor.predict(test_data.drop(columns=[target]))

# Evaluate accuracy
performance = predictor.evaluate(test_data)
print(f"Model Accuracy: {performance['accuracy']:.2f}")


leaderboard = predictor.leaderboard(test_data)
print(leaderboard)



new_passenger = pd.DataFrame({
    'PassengerId': [99999],  
    'Pclass': [3],           # Passenger class (1st, 2nd, 3rd)
    'Name': ['John Doe'],   
    'Sex': ['male'],        
    'Age': [25],            
    'Ticket': ['UNKNOWN'],  
    'Fare': [7.25],         
    'Cabin': ['UNKNOWN'],  
    'Embarked': ['S'],      # Most common value ('S' for Southampton)
    'SibSp': [0],           # Siblings aboard
    'Parch': [0]            # Parents/children aboard
})

prediction = predictor.predict(new_passenger)
print(f"Survival prediction: {'Yes' if prediction[0] == 1 else 'No'}")