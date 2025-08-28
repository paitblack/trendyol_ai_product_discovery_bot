import os
import json
from pathlib import Path

def count_links_in_json_files(root_directory):
    total_links = 0
    processed_files = 0
    error_files = []
    
    root_path = Path(root_directory)
    
    json_files = list(root_path.rglob('*.json'))
    
    print(f"Total {len(json_files)} JSON file found.")
    print("-" * 50)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                if isinstance(data, list):
                    links_in_file = len(data)
                    total_links += links_in_file
                    print(f"{json_file.relative_to(root_path)}: {links_in_file} link")
                
                elif isinstance(data, dict) and 'links' in data:
                    links_in_file = len(data['links'])
                    total_links += links_in_file
                    print(f"{json_file.relative_to(root_path)}: {links_in_file} link")
                
                else:
                    links_in_file = 1 if 'link' in str(data) else 0
                    total_links += links_in_file
                    print(f"{json_file.relative_to(root_path)}: {links_in_file} link")
                
                processed_files += 1
                
        except json.JSONDecodeError:
            error_files.append(str(json_file))
            print(f"err: {json_file.relative_to(root_path)}")
            
        except Exception as e:
            error_files.append(str(json_file))
            print(f"err: {json_file.relative_to(root_path)} - {str(e)}")
    
    print("-" * 50)
    print(f"results:")
    print(f"files done: {processed_files}")
    print(f"error files: {len(error_files)}")
    print(f"total link: {total_links}")
    
    if error_files:
        print("error files:")
        for error_file in error_files:
            print(f"  - {error_file}")
    
    return {
        'total_links': total_links,
        'processed_files': processed_files,
        'error_files': error_files,
        'total_json_files': len(json_files)
    }

if __name__ == "__main__":
    root_directory = input("C:/Users/emre-/Desktop/mava/Trendyol").strip()
    
    if not root_directory:
        root_directory = "."
    
    if not os.path.exists(root_directory):
        print(f"err: '{root_directory}' path not find")
    else:
        print(f"'{root_directory}' scanning")
        results = count_links_in_json_files(root_directory)
        
        print(f"summ: {results['total_json_files']} json files {results['total_links']} link found")