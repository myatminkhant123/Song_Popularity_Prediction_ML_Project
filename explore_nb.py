import pandas as pd

# Load both datasets
spotify = pd.read_csv('SpotifyFeatures.csv')
hot100 = pd.read_csv('Hot100.csv')

print("=== SPOTIFY FEATURES ===")
print(f"Shape: {spotify.shape}")
print(f"Columns: {list(spotify.columns)}")
print(f"\nSample:\n{spotify.head(3).to_string()}")
print(f"\nNull values:\n{spotify.isnull().sum()}")
print(f"\nData types:\n{spotify.dtypes}")

print("\n\n=== HOT 100 ===")
print(f"Shape: {hot100.shape}")
print(f"Columns: {list(hot100.columns)}")
print(f"\nSample:\n{hot100.head(3).to_string()}")
print(f"\nNull values:\n{hot100.isnull().sum()}")
print(f"\nData types:\n{hot100.dtypes}")
print(f"\nPopularity stats:\n{hot100['Popularity'].describe()}")
