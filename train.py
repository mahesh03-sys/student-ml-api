from sklearn.linear_model import LinearRegression
import joblib

# Data: [study_hours, attendance_percentage, prev_marks]
X = [
    [2, 60, 50], 
    [4, 70, 60], 
    [6, 80, 70], 
    [8, 90, 80], 
    [10, 95, 90]
]
# Target: performance_score
y = [55, 65, 75, 85, 95] 

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model trained successfully! Saved as model.pkl")