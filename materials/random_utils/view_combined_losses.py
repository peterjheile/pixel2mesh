#######################################################################################
#
#   I used this page to help create a nice visualiztion for each of teh loss functions combined
#   For my poster. I used chat gpt to help with a bulk of this logic.
#   As far os the model goes THERE IS NO LOGIC here that is important for understanding my implemntatiuon
#
#
#######################################################################################



import pandas as pd
import glob

#these are the losses for 
csv_files = {
    "Chamfer": "chamfer.csv",
    "Normal": "normal.csv",
    "Edge": "edge.csv",
    "Laplacian": "laplacian.csv",
    "Total": "total.csv"
}


dataframes = []

for name, path in csv_files.items():
    df = pd.read_csv(path)
    df = df[["Step", "Value"]]
    df = df.rename(columns={"Value": name})
    dataframes.append(df)

# === Step 3: Merge on 'Step'
merged = dataframes[0]
for df in dataframes[1:]:
    merged = pd.merge(merged, df, on="Step", how="outer")

# === Step 4: Sort and save
merged = merged.sort_values("Step").reset_index(drop=True)
merged.to_csv("combined_losses.csv", index=False)
print("✅ Combined CSV saved as: combined_losses.csv")


# Load original combined loss CSV
df = pd.read_csv("combined_losses.csv")

# --- Step 1: Apply smoothing (moving average)
window_size = 10  # adjust as needed
smoothed_df = df.copy()
for col in df.columns[1:]:  # skip 'Step'
    smoothed_df[col] = df[col].rolling(window=window_size, min_periods=1).mean()

# --- Step 2: Downsample (keep every 100th row)
downsampled_df = smoothed_df.iloc[::10].reset_index(drop=True)

# --- Step 3: Save the result
downsampled_df.to_csv("smoothed_downsampled_losses.csv", index=False)
print("✅ Saved cleaned CSV as: smoothed_downsampled_losses.csv")


import pandas as pd

df = pd.read_csv("smoothed_downsampled_losses.csv")

# Normalize each column (except 'Step')
normalized_df = df.copy()
for col in df.columns[1:]:
    col_min = df[col].min()
    col_max = df[col].max()
    normalized_df[col] = (df[col] - col_min) / (col_max - col_min)

normalized_df.to_csv("normalized_losses.csv", index=False)
print("✅ Normalized losses saved as: normalized_losses.csv")