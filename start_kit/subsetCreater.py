import json

# Define the exact list of 56 glosses for the cafe conversation subset
target_glosses = [
    # Greetings & Politeness
    "hello", "thank you", "please", "yes", "no", "sorry", "good", "finish",
    
    # Pronouns & Core Verbs
    "I", "me", "you", "want", "have", "need", "like", "will", "can", "help", 
    "pay", "make", "go", "work",
    
    # Food & Drink
    "coffee", "tea", "water", "milk", "sugar", "food", "drink", "sandwich", 
    "cake", "cookie", "chocolate", "eat", "hot", "cold",
    
    # Objects & Concepts
    "table", "menu", "money", "credit card", "bill", "name", "time", "bathroom", "toilet",
    
    # Questions & Descriptors
    "what", "where", "how", "how much", "more", "again", "same", "big", "small", "ready"
]

# Define file paths
main_json_path = 'WLASL_v0.3.json'
subset_json_path = 'WLASL_Cafe.json'

cafe_subset = []

print(f"Loading the full dataset from {main_json_path}...")
# Load the entire dataset from the JSON file
with open(main_json_path, 'r') as f:
    full_dataset = json.load(f)

print("Searching for your target words...")
# Iterate through the full dataset to find the target glosses
for entry in full_dataset:
    if entry['gloss'] in target_glosses:
        cafe_subset.append(entry)

# --- Saving the results ---
print(f"Found {len(cafe_subset)} out of {len(target_glosses)} requested words.")
print(f"Saving the custom subset to {subset_json_path}...")

with open(subset_json_path, 'w') as f:
    json.dump(cafe_subset, f, indent=4)

print("✅ Done! Your 'WLASL_Cafe.json' file has been created.")