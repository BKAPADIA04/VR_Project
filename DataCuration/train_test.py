import pandas as pd
path = pd.read_csv("./abo-images-small/images/metadata/images.csv")
import json
import pandas as pd

# Load your JSON file
with open('./merge_final.json', 'r') as f:
    data = json.load(f)

# Flatten the data
rows = []
for item in data:
    item_id = item.get('item_id')
    image_id = item.get('image_id')
    for qa in item.get('questions', []):
        rows.append({
            'item_id': item_id,
            'image_id': image_id,
            'question': qa.get('question'),
            'answer': qa.get('answer')
        })

# Convert to DataFrame
df = pd.DataFrame(rows)
print('Length: ', len(df))
import pandas as pd

# Assuming df1 and df2 are already defined:
# df1: item_id, image_id, question, answer
# df2: image_id, height, width, path

# Merge on 'image_id'
merged_df = pd.merge(df, path[['image_id', 'path']], on='image_id', how='inner')

# Select desired columns
final_df = merged_df[['image_id','question', 'answer', 'path']]

# Display or save result

print('Final Dataframe Length: ', final_df)

from sklearn.model_selection import train_test_split

# Split the dataframe
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42, shuffle=True)

# Save to CSV
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print(f"Train dataset saved with {len(train_df)} entries")
print(f"Test dataset saved with {len(test_df)} entries")


