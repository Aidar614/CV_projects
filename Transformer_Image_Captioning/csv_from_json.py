import json
import csv

json_file = 'annotations_trainval2017/annotations/captions_val2017.json'

with open(json_file, 'r') as f:
    data = json.load(f)

with open('captions_val.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "caption"])  

    for item in data['annotations']:
        image_id = item['image_id']
        file_name = f"{str(image_id).zfill(12)}.jpg"
        image_path = f"val2017/val2017/{file_name}"
        caption = item['caption']
        writer.writerow([image_path, caption]) 

print("CSV файл image_captions.csv создан!")