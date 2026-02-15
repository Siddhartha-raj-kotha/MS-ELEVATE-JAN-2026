import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
import os

# 1. Load data
df = pd.read_csv('data/diet_data.csv')

# 2. Setup AI Features (Protein and Cost)
X = df[['Protein', 'Cost']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train KNN Model
model = NearestNeighbors(n_neighbors=3, metric='euclidean')
model.fit(X_scaled)

# 4. Save the "Brain" into the models folder
with open('models/fitness_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ AI Training Complete! Files saved in 'models/' folder.")