import json
import os

def load_data(index_file_path):
    """
    Loads the WLASL dataset JSON file and returns a flat list of 
    all video instances, each represented as a dictionary.

    Args:
        index_file_path (str): The path to the WLASL JSON file 
                               (e.g., 'WLASL_Cafe.json').

    Returns:
        list: A list of dictionaries, where each dictionary represents 
              a single video instance and contains keys like 'video_id', 
              'gloss', 'split', 'url', etc. 
              Returns an empty list if the file cannot be read.
    """
    try:
        with open(index_file_path, 'r') as f:
            content = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{index_file_path}'.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{index_file_path}'. Check file validity.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred reading {index_file_path}: {e}")
        return []

    all_instances = []
    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']
        for inst in instances:
            # Create a dictionary for this instance, copying relevant keys
            instance_data = {
                'gloss': gloss,
                'video_id': inst.get('video_id'), # Use .get() for safety
                'split': inst.get('split'),
                'url': inst.get('url'),
                'frame_start': inst.get('frame_start'),
                'frame_end': inst.get('frame_end'),
                # Add any other keys you might need later
            }
            all_instances.append(instance_data)
            
    print(f"Loaded {len(all_instances)} video instances from {index_file_path}")
    return all_instances

# --- Example Usage (Optional - for testing the script directly) ---
# This part only runs if you execute 'python data_reader.py' directly
if __name__ == '__main__':
    # --- IMPORTANT: Change this to your actual JSON file ---
    test_json_file = 'WLASL_Cafe.json' 
    # --- Or 'WLASL100.json' / 'WLASL_v0.3.json' ---

    if os.path.exists(test_json_file):
        data_list = load_data(test_json_file)
        if data_list:
            print("\nSuccessfully loaded data.")
            print(f"Total instances found: {len(data_list)}")
            print("\nExample of the first instance:")
            print(data_list[0]) 
            
            # Example: Create the lookup dictionary 
            video_to_label_split = {}
            for instance in data_list:
                video_id = instance['video_id']
                gloss = instance['gloss']
                split = instance['split']
                if video_id: # Make sure video_id exists
                    video_to_label_split[video_id] = {'gloss': gloss, 'split': split}
            
            print(f"\nCreated lookup dictionary with {len(video_to_label_split)} entries.")
            # Print one example from the dictionary
            example_id = data_list[0].get('video_id')
            if example_id:
                 print(f"Example lookup for video ID '{example_id}': {video_to_label_split.get(example_id)}")
    else:
        print(f"Error: Test JSON file '{test_json_file}' not found in the current directory.")
        print("Please create the file or change the 'test_json_file' variable.")