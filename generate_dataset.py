import pandas as pd
import numpy as np

# Generate data
rows = 20
cols = 5
column_names = [f"Column_{i+1}" for i in range(cols)]

data = np.random.randint(1, 100, size=(rows, cols))

df = pd.DataFrame(data, columns=column_names)

# Save as CSV
df.to_csv("sample_dataset.csv", index=False)

print("CSV file 'sample_dataset.csv' generated successfully!")