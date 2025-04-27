# player_cluster.py
import cv2
import numpy as np
from collections import Counter

class PlayerCluster:
    def assign_clusters(self, cluster_labels, cluster_history, assigned_clusters, threshold=5):
        updated_assigned_clusters = assigned_clusters.copy()

        for tracker_id, current_cluster in cluster_labels:
            tracker_id = int(tracker_id)
            if current_cluster is None:
                continue
            current_cluster = int(current_cluster)

            if tracker_id not in cluster_history:
                cluster_history[tracker_id] = Counter()
            cluster_history[tracker_id][current_cluster] += 1
            
            cluster_history[tracker_id] = +cluster_history[tracker_id]
            best_cluster = cluster_history[tracker_id].most_common(1)

            if best_cluster and best_cluster[0][1] >= threshold:
                updated_assigned_clusters[tracker_id] = best_cluster[0][0]
            elif tracker_id in updated_assigned_clusters and (not best_cluster or best_cluster[0][1] < threshold):
                del updated_assigned_clusters[tracker_id]

        return updated_assigned_clusters

    def cluster_players(self, img, detections, names, clustering_model):
        player_crops = []
        track_ids = []
        for xyxy, confidence, class_id, track_id in zip(detections.xyxy, detections.confidence, 
                                                      detections.class_id, detections.tracker_id):
            if names[class_id] == 'player':
                x1, y1, x2, y2 = map(int, xyxy)
                player_crops.append(img[y1:y2, x1:x2])
                track_ids.append(track_id)

        if not player_crops:
            return []

        def process_crop(crop):
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv_crop, np.array([35, 40, 40]), np.array([90, 255, 255]))
            masked_crop = cv2.bitwise_and(crop, crop, mask=~mask)
            flat_crop = masked_crop.reshape(-1, 3)[np.any(masked_crop.reshape(-1, 3) != 0, axis=1)]
            return flat_crop

        flat_crops = list(map(process_crop, player_crops))

        if not flat_crops:
            return []

        flat_combined = np.vstack(flat_crops)
        labels = clustering_model.fit_predict(flat_combined)

        img_labels = []
        start = 0
        for i in range(len(player_crops)):
            num_pixels = len(flat_crops[i])
            if num_pixels == 0:
                img_labels.append(None)
            else:
                cluster_mask = labels[start:start + num_pixels]
                counts = np.bincount(cluster_mask)
                img_labels.append(np.argmax(counts) if len(counts) > 1 else cluster_mask[0])
            start += num_pixels

        player_clusters = list(zip(track_ids, img_labels))
        return player_clusters