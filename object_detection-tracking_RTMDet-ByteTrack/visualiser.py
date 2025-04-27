import cv2

def BboxCorner(frame, x1, y1, x2, y2, color= (255, 0,0)):
    
    width = x2 - x1
    height = y2 - y1

    gap_size_width = int(width * 0.7)
    gap_size_height = int(height * 0.7)

   
    cv2.line(frame, (x1, y1), (x1 + (width - gap_size_width) // 2, y1), color, 2)
    cv2.line(frame, (x2 - (width - gap_size_width) // 2, y1), (x2, y1), color, 2)

    
    cv2.line(frame, (x1, y2), (x1 + (width - gap_size_width) // 2, y2), color, 2)
    cv2.line(frame, (x2 - (width - gap_size_width) // 2, y2), (x2, y2), color, 2)

    
    cv2.line(frame, (x1, y1), (x1, y1 + (height - gap_size_height) // 2), color, 2)
    cv2.line(frame, (x1, y2 - (height - gap_size_height) // 2), (x1, y2), color, 2)

   
    cv2.line(frame, (x2, y1), (x2, y1 + (height - gap_size_height) // 2), color, 2)
    cv2.line(frame, (x2, y2 - (height - gap_size_height) // 2), (x2, y2), color, 2)
    
    return frame


def visualizer(frames, detections):
    
    annotated_images = []
    for i in range(len(frames)):
        frame = frames[i]  
        frame_detections = detections[i]  
        
        detection = frame_detections.xyxy
        confidence = frame_detections.confidence
        class_ids = frame_detections.class_id

        person_count = 0
        motorcycle_count = 0

        for j in range(len(detection)):
            bbox = detection[j]
            conf = confidence[j]
            class_id = class_ids[j]

            x1, y1, x2, y2 = map(int, bbox)

            color = (0, 255, 0)  
            if class_id in coco_classes:
                class_name = coco_classes[class_id]['name']
                
                color = coco_classes[class_id]['color'] if class_id in coco_classes else (0, 255, 0)
                
                if class_name == 'person':
                    frame = BboxCorner(frame, x1, y1, x2, y2, color)
                    person_count += 1
                else:
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    motorcycle_count += 1

            
            text = f"{class_name} Conf: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        
        text_motorcycles = f"Motorcycles: {motorcycle_count}"
        
        text_size_motorcycles, _ = cv2.getTextSize(text_motorcycles, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x_pos = 10  
        y_pos = frame.shape[0] - 50  

        
        cv2.putText(frame, text_motorcycles, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (122, 255, 165), 2)

        annotated_images.append(frame)

    return annotated_images




coco_classes = {
    0: {'name': 'person', 'color': (255, 30, 255)},  
    1: {'name': 'bicycle', 'color': (0, 255, 0)},  # Зеленый
    2: {'name': 'car', 'color': (30, 30, 255)}, 
    3: {'name': 'motorcycle', 'color': (255, 55, 0)},  
    4: {'name': 'airplane', 'color': (0, 255, 255)},  # Голубой
    5: {'name': 'bus', 'color': (255, 0, 255)},  # Пурпурный
    6: {'name': 'train', 'color': (128, 0, 0)},  # Темно-красный
    7: {'name': 'truck', 'color': (0, 128, 0)},  # Темно-зеленый
    8: {'name': 'boat', 'color': (0, 0, 128)},  # Темно-синий
    9: {'name': 'traffic light', 'color': (128, 128, 0)},  # Оливковый
    10: {'name': 'fire hydrant', 'color': (0, 128, 128)},  # Бирюзовый
    11: {'name': 'stop sign', 'color': (128, 0, 128)},  # Фиолетовый
    12: {'name': 'parking meter', 'color': (192, 192, 192)},  # Светло-серый
    13: {'name': 'bench', 'color': (255, 165, 0)},  # Оранжевый
    14: {'name': 'bird', 'color': (255, 192, 203)},  # Розовый
    15: {'name': 'cat', 'color': (139, 69, 19)},  # Коричневый
    16: {'name': 'dog', 'color': (255, 215, 0)},  # Золотой
    17: {'name': 'horse', 'color': (128, 128, 128)},  # Серый
    18: {'name': 'sheep', 'color': (255, 255, 255)},  # Белый
    19: {'name': 'cow', 'color': (0, 0, 0)},  # Черный
    20: {'name': 'elephant', 'color': (255, 140, 0)},  # Темно-оранжевый
    21: {'name': 'bear', 'color': (128, 0, 128)},  # Фиолетовый
    22: {'name': 'zebra', 'color': (0, 128, 128)},  # Бирюзовый
    23: {'name': 'giraffe', 'color': (128, 128, 128)},  # Серый
    24: {'name': 'backpack', 'color': (255, 165, 0)},  # Оранжевый
    25: {'name': 'umbrella', 'color': (255, 192, 203)},  # Розовый
    26: {'name': 'handbag', 'color': (139, 69, 19)},  # Коричневый
    27: {'name': 'tie', 'color': (255, 215, 0)},  # Золотой
    28: {'name': 'suitcase', 'color': (128, 0, 0)},  # Темно-красный
    29: {'name': 'frisbee', 'color': (0, 128, 0)},  # Темно-зеленый
    30: {'name': 'skis', 'color': (0, 0, 128)},  # Темно-синий
    31: {'name': 'snowboard', 'color': (128, 128, 0)},  # Оливковый
    32: {'name': 'sports ball', 'color': (128, 0, 128)},  # Фиолетовый
    33: {'name': 'kite', 'color': (0, 128, 128)},  # Бирюзовый
    34: {'name': 'baseball bat', 'color': (128, 128, 128)},  # Серый
    35: {'name': 'baseball glove', 'color': (255, 165, 0)},  # Оранжевый
    36: {'name': 'skateboard', 'color': (255, 192, 203)},  # Розовый
    37: {'name': 'surfboard', 'color': (139, 69, 19)},  # Коричневый
    38: {'name': 'tennis racket', 'color': (255, 215, 0)},  # Золотой
    39: {'name': 'bottle', 'color': (128, 0, 0)},  # Темно-красный
    40: {'name': 'wine glass', 'color': (0, 128, 0)},  # Темно-зеленый
    41: {'name': 'cup', 'color': (0, 0, 128)},  # Темно-синий
    42: {'name': 'fork', 'color': (128, 128, 0)},  # Оливковый
    43: {'name': 'knife', 'color': (128, 0, 128)},  # Фиолетовый
    44: {'name': 'spoon', 'color': (0, 128, 128)},  # Бирюзовый
    45: {'name': 'bowl', 'color': (128, 128, 128)},  # Серый
    46: {'name': 'banana', 'color': (255, 165, 0)},  # Оранжевый
    47: {'name': 'apple', 'color': (255, 192, 203)},  # Розовый
    48: {'name': 'sandwich', 'color': (139, 69, 19)},  # Коричневый
    49: {'name': 'orange', 'color': (255, 215, 0)},  # Золотой
    50: {'name': 'broccoli', 'color': (128, 0, 0)},  # Темно-красный
    51: {'name': 'carrot', 'color': (0, 128, 0)},  # Темно-зеленый
    52: {'name': 'hot dog', 'color': (0, 0, 128)},  # Темно-синий
    53: {'name': 'pizza', 'color': (128, 128, 0)},  # Оливковый
    54: {'name': 'donut', 'color': (128, 0, 128)},  # Фиолетовый
    55: {'name': 'cake', 'color': (0, 128, 128)},  # Бирюзовый
    56: {'name': 'chair', 'color': (128, 128, 128)},  # Серый
    57: {'name': 'couch', 'color': (255, 165, 0)},  # Оранжевый
    58: {'name': 'potted plant', 'color': (255, 192, 203)},  # Розовый
    59: {'name': 'bed', 'color': (139, 69, 19)},  # Коричневый
    60: {'name': 'dining table', 'color': (255, 215, 0)},  # Золотой
    61: {'name': 'toilet', 'color': (128, 0, 0)},  # Темно-красный
    62: {'name': 'tv', 'color': (0, 128, 0)},  # Темно-зеленый
    63: {'name': 'laptop', 'color': (0, 0, 128)},  # Темно-синий
    64: {'name': 'mouse', 'color': (128, 128, 0)},  # Оливковый
    65: {'name': 'remote', 'color': (128, 0, 128)},  # Фиолетовый
    66: {'name': 'keyboard', 'color': (0, 128, 128)},  # Бирюзовый
    67: {'name': 'cell phone', 'color': (128, 128, 128)},  # Серый
    68: {'name': 'microwave', 'color': (255, 165, 0)},  # Оранжевый
    69: {'name': 'oven', 'color': (255, 192, 203)},  # Розовый
    70: {'name': 'toaster', 'color': (139, 69, 19)},  # Коричневый
    71: {'name': 'sink', 'color': (255, 215, 0)},  # Золотой
    72: {'name': 'refrigerator', 'color': (128, 0, 0)},  # Темно-красный
    73: {'name': 'book', 'color': (0, 128, 0)},  # Темно-зеленый
    74: {'name': 'clock', 'color': (0, 0, 128)},  # Темно-синий
    75: {'name': 'vase', 'color': (128, 128, 0)},  # Оливковый
    76: {'name': 'scissors', 'color': (128, 0, 128)},  # Фиолетовый
    77: {'name': 'teddy bear', 'color': (0, 128, 128)},  # Бирюзовый
    78: {'name': 'hair drier', 'color': (128, 128, 128)},  # Серый
    79: {'name': 'toothbrush', 'color': (255, 165, 0)}  # Оранжевый
}