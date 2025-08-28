import os
import json
from pathlib import Path

def merge_json_files(root_directory, output_file):
    merged_data = []

    root_path = Path(root_directory)

    json_files = list(root_path.rglob('*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    merged_data.extend(data)
                elif isinstance(data, dict):
                    merged_data.append(data)
        except json.JSONDecodeError:
            print(f"d . error")
        except Exception as e:
            print(f"Eerr")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    
    #root_directory = "C:/Users/emre-/Desktop/mava/Trendyol"
    #output_file = "C:/Users/emre-/Desktop/mava/trendyol.json"

    #if not os.path.exists(root_directory):
    #    print(f"Error: '{root_directory}' path not found")
    #else:
    #    merge_json_files(root_directory, output_file)
    #    print(f"Merged JSON files into {output_file}")

    with open('C:/Users/emre-/Desktop/mava/trendyol.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(len(data))
        