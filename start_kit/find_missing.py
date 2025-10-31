import os
import json
from collections import defaultdict

# --- IMPORTANT: Make sure this points to your custom JSON file ---
json_path = 'WLASL_Cafe.json' 
# --- Or use 'WLASL100.json' or 'WLASL_v0.3.json' if needed ---

# Directory where raw videos are downloaded
video_dir = 'videos'

# Output file for the detailed list of missing videos
missing_file_path = 'missing.txt'

def find_missing_videos(index_file, raw_video_dir):
    """
    Finds missing videos, writes details to a file, and prints a summary 
    including available counts per gloss and total counts.
    """
    try:
        with open(index_file, 'r') as f:
            content = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{index_file}'. Please check the path.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{index_file}'. Check if the file is valid.")
        return

    missing_videos_details = []
    video_status_per_gloss = defaultdict(lambda: {'total': 0, 'missing': 0}) 
    total_videos_checked = 0
    
    print(f"Checking for videos listed in '{index_file}' inside the '{raw_video_dir}' folder...")

    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']
        
        for inst in instances:
            total_videos_checked += 1
            video_id = inst['video_id']
            video_status_per_gloss[gloss]['total'] += 1 # Increment total count for this gloss
            
            # Construct the expected path (checking for common extensions)
            possible_filenames = [
                f"{video_id}.mp4", 
                f"{video_id}.mkv", 
                f"{video_id}.swf", 
                inst['url'][-11:] + '.mp4', # Handle YouTube IDs as filenames
                inst['url'][-11:] + '.mkv'  # Handle YouTube IDs as filenames
            ]
            
            found = False
            for filename in possible_filenames:
                savepath = os.path.join(raw_video_dir, filename)
                if os.path.exists(savepath):
                    found = True
                    break # Found one version, no need to check others
            
            # If none of the possible filenames exist, mark as missing
            if not found:
                missing_videos_details.append({'id': video_id, 'gloss': gloss})
                video_status_per_gloss[gloss]['missing'] += 1 # Increment missing count for this gloss

    # --- Write the detailed missing list to file ---
    with open(missing_file_path, 'w') as f:
        f.write("Missing Video ID\tGloss\n") # Header line
        f.write("-----------------\t-----\n")
        for item in missing_videos_details:
            f.write(f"{item['id']}\t{item['gloss']}\n")
            
    # --- Print the summary to the console ---
    print("\n--- Missing Videos Summary ---")
    total_glosses = len(content) # Get the total number of glosses
    total_missing = len(missing_videos_details)

    if total_missing == 0:
        print("🎉 No missing videos found!")
    else:
        print(f"Checked {total_glosses} glosses and {total_videos_checked} total video instances.")
        print(f"Found {total_missing} missing videos.")
        print(f"Detailed list saved to '{missing_file_path}'")
        print("\nStatus per gloss (Missing, Available / Total):")
        # Sort by gloss name for readability
        for gloss, counts in sorted(video_status_per_gloss.items()):
            available_count = counts['total'] - counts['missing']
            print(f"- {gloss}: {counts['missing']} missing, {available_count} available / {counts['total']} total")
    
    print("\n--- Totals ---")
    print(f"Total Glosses Checked: {total_glosses}")
    print(f"Total Videos Missing: {total_missing}")
    print("--------------")


if __name__ == '__main__':
    if not os.path.exists(video_dir):
        print(f"Error: Raw videos directory '{video_dir}' not found.")
        print("Please make sure you have run the video downloader first.")
    else:
        find_missing_videos(json_path, video_dir)