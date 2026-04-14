import os 
import pandas as pd

path = "./exports"
folders = ["013", "015", "08"]
all_data = []

for folder in folders:
    folder_path = os.path.join(path, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            parts = file.replace(".csv", "").split("_")
            label = parts[0]
            img_id = parts[1]
            sample_id = parts[2]
            
            df = pd.read_csv(os.path.join(folder_path, file))
            df['class'] = label
            df['source_image'] = img_id
            df['sample_id'] = sample_id + img_id
            
            all_data.append(df)

master_df = pd.concat(all_data, ignore_index=True)
master_df.to_csv("spectral_library_master.csv", index=False)