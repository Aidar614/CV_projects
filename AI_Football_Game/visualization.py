import cv2
import numpy as np
import supervision as sv
from collections import deque

class Visualization:
    def __init__(self):
        self.class_colors = {
            'ball': (255, 0, 125),  # BGR format
            'goalkeeper': (0, 255, 0),
            'player': (210, 50, 255),
            'referee': (255, 255, 0)
        }
        
        self.ball_trajectory = deque(maxlen=120) 
        self.min_thickness = 2  
        
    def update_ball_trajectory(self, ball_coords):
        if ball_coords is not None:
           
            self.ball_trajectory.append((ball_coords[0], ball_coords[1], 1.0))
        
        for i in range(len(self.ball_trajectory)):
            x, y, alpha = self.ball_trajectory[i]
    
            new_alpha = max(0, alpha - 0.02)
            self.ball_trajectory[i] = (x, y, new_alpha)
        
        while len(self.ball_trajectory) > 0 and self.ball_trajectory[0][2] <= 0:
            self.ball_trajectory.popleft()

    def overlay_image(self, frame_with_boxes, image, scale_factor, alpha, x_offset=-10, y_offset=-35):
        new_size = (int(image.shape[1] * scale_factor), 
                    int(image.shape[0] * scale_factor))
        resized_image = cv2.resize(image, new_size)
        
        overlay = resized_image.copy()
        mask = np.ones(overlay.shape, dtype=np.float32) * alpha
        
        h, w = frame_with_boxes.shape[:2]
        oh, ow = overlay.shape[:2]
        
        x = w - ow
        y = h - oh
        
        x += x_offset 
        y += y_offset  
        
        x = max(0, x)
        y = max(0, y)
        
        if x + ow > w:
            ow = w - x
            overlay = overlay[:, :ow]
        if y + oh > h:
            oh = h - y
            overlay = overlay[:oh, :]
        
        roi = frame_with_boxes[y:y + oh, x:x + ow]
        if roi.shape == overlay.shape:
            frame_with_boxes[y:y + oh, x:x + ow] = cv2.addWeighted(
                roi, 1-alpha, overlay.astype(roi.dtype), alpha, 0)
        
        return frame_with_boxes

    def draw_football_pitch(self, vertices, transformed_player_points, 
                        names, assigned_clusters, class_colors):
    
        width, height = 1200, 800  
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        image[:] = (50, 180, 50)
        
        def scale_coords(x, y):
            pitch_length = 120  
            pitch_width = 80  
            scale_x = width / pitch_length
            scale_y = height / pitch_width
            return int(x * scale_x), int(y * scale_y)
        
        def draw_line(x1, y1, x2, y2, thickness=2):
            x1, y1 = scale_coords(x1, y1)
            x2, y2 = scale_coords(x2, y2)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        
        # Основные линии поля
        draw_line(0, 0, 120, 0)  # верхняя линия
        draw_line(0, 80, 120, 80)  # нижняя линия
        draw_line(0, 0, 0, 80)  # левая линия
        draw_line(120, 0, 120, 80)  # правая линия
        draw_line(60, 0, 60, 80)  # центральная линия
        
        # Центральный круг
        center_x, center_y = scale_coords(60, 40)
        cv2.circle(image, (center_x, center_y), scale_coords(10, 0)[0], (255, 255, 255), 2)
        
        # Штрафные площади
        draw_line(0, (80-43)/2, 17.8, (80-43)/2)  # левая верхняя
        draw_line(17.8, (80-43)/2, 17.8, (80+43)/2)  # левая вертикальная
        draw_line(0, (80+43)/2, 17.8, (80+43)/2)  # левая нижняя
        
        draw_line(120, (80-43)/2, 120-17.8, (80-43)/2)  # правая верхняя
        draw_line(120-17.8, (80-43)/2, 120-17.8, (80+43)/2)  # правая вертикальная
        draw_line(120, (80+43)/2, 120-17.8, (80+43)/2)  # правая нижняя
        
        # Вратарские площади
        draw_line(0, (80-19)/2, 5.5, (80-19)/2)  # левая верхняя
        draw_line(5.5, (80-19)/2, 5.5, (80+19)/2)  # левая вертикальная
        draw_line(0, (80+19)/2, 5.5, (80+19)/2)  # левая нижняя
        
        draw_line(120, (80-19)/2, 120-5.5, (80-19)/2)  # правая верхняя
        draw_line(120-5.5, (80-19)/2, 120-5.5, (80+19)/2)  # правая вертикальная
        draw_line(120, (80+19)/2, 120-5.5, (80+19)/2)  # правая нижняя
        
        
        # Точки для пробития пенальти
        penalty_spot_left = scale_coords(12, 40)
        penalty_spot_right = scale_coords(120-12, 40)
        cv2.circle(image, penalty_spot_left, 5, (255, 255, 255), -1)
        cv2.circle(image, penalty_spot_right, 5, (255, 255, 255), -1)
        
       
        for i, (x, y) in enumerate(vertices):
            px, py = scale_coords(x, y)
            cv2.circle(image, (px, py), 5, (255, 255, 255), -1)
            
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x = px + 5  
            text_y = py + 5
            
            if text_x + text_width > width:
                text_x = px - text_width - 5  
                
            if text_y + text_height > height:
                text_y = py - text_height - 5 
                
            if text_x < 0:
                text_x = px + 5
                
            if text_y < 0:
                text_y = py + 5
                           
            cv2.putText(image, text, (text_x, text_y), font, 
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        if len(self.ball_trajectory) >= 2:
            trajectory_img = np.zeros_like(image, dtype=np.uint8)
            
            base_color = self.class_colors['ball'] 
            fixed_alpha = 1 
            
            for i in range(1, len(self.ball_trajectory)):
                x1, y1, alpha1 = self.ball_trajectory[i-1]
                x2, y2, alpha2 = self.ball_trajectory[i]
                
                px1, py1 = scale_coords(x1, y1)
                px2, py2 = scale_coords(x2, y2)
                
                thickness = max(self.min_thickness, int(5 * ((alpha1 + alpha2) / 2)))
                
                cv2.line(trajectory_img, (px1, py1), (px2, py2), base_color, thickness)
            
            mask = np.zeros_like(image, dtype=np.float32)
            for i in range(1, len(self.ball_trajectory)):
                x1, y1, alpha1 = self.ball_trajectory[i-1]
                x2, y2, alpha2 = self.ball_trajectory[i]
                
                px1, py1 = scale_coords(x1, y1)
                px2, py2 = scale_coords(x2, y2)
                
                cv2.line(mask, (px1, py1), (px2, py2), (fixed_alpha, fixed_alpha, fixed_alpha), thickness)
            
            image = cv2.addWeighted(image, 1.0, trajectory_img, fixed_alpha, 0)
        
        ball_coords = None
        if transformed_player_points is not None:
            
            
            for track_id, coords, class_id in transformed_player_points:
                px, py = scale_coords(coords[0], coords[1])
                class_name = names.get(int(class_id), 'unknown')
                
                if class_name == 'ball':
                    ball_coords = (coords[0], coords[1])
                
                color = class_colors.get(class_name, (170, 50, 50))  
                
                if isinstance(color, tuple) and len(color) == 3:
                    color = (color[2], color[1], color[0])
                
                if class_name == 'player' and track_id in assigned_clusters:
                    cluster = assigned_clusters[track_id]
                    if cluster == 0:
                        color = (255, 0, 0)  
                    elif cluster == 1:
                        color = (0, 0, 255)  
                
                cv2.circle(image, (px, py), 8, color, -1)
        
        self.update_ball_trajectory(ball_coords)
        
        return image

    def draw_boxes(self, image, detections: sv.Detections, assigned_clusters, names, class_colors):
        ellipse_axes_x = 30
        ellipse_axes_y = 15

        for i, (xyxy, confidence, class_id, track_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id)):
            x1, y1, x2, y2 = map(int, xyxy)
            class_name = names[class_id]
            
            color = (170, 50, 50)  
            
            if class_name in class_colors:
                color = class_colors[class_name]
            
            if class_name == 'player' and track_id in assigned_clusters:
                cluster = assigned_clusters[track_id]
                if cluster == 0:
                    color = (255, 0, 0)  
                elif cluster == 1:
                    color = (0, 0, 255)  
            
            if class_name == 'ball':
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                
                cv2.circle(image, (center_x, center_y), radius, color, 2)
                
                label = f"ball: {confidence:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y - radius - 10 if (center_y - radius - 10) > 0 else center_y + radius + 20
                cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                continue  
            
            center_x = (x1 + x2) // 2
            center_y = y2 + ellipse_axes_y // 2 
            
            cv2.ellipse(image, (center_x, center_y), (ellipse_axes_x, ellipse_axes_y), 0, 0, 180, color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            if track_id is not None:
                label = f"{class_name} ({track_id}): {confidence:.2f}"
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1
            text_y = y1 - text_size[1] - 5 if y1 - text_size[1] - 5 > 0 else y1 + text_size[1] + 5
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image