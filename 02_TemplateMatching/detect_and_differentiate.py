"""
Automatic Detection and Differentiation Algorithm for Hanger and Connector Objects

Multi-stage detection pipeline:
1. Multi-scale Template Matching (TM_CCOEFF_NORMED) at 25 scales (0.3-1.5x)
2. Local Maximum Extraction via morphological dilation (3x3 kernel)
3. Non-Maximum Suppression using IoU thresholding (Hanger: 0.5, Connector: 0.4)
4. SIFT Feature Verification with Lowe's ratio test (min_matches=3)
5. ORB Fallback for sparse features (when SIFT keypoints < threshold)

Thresholds:
- Hanger: match_threshold=0.45 (distinctive features)
- Connector: match_threshold=0.33 (subtle variations)
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os


class HangerConnectorDetector:
    """
    Multi-stage detector for Hanger and Connector objects using template matching and SIFT verification.
    
    Pre-computes SIFT/ORB features from templates for efficient multi-image detection.
    """
    
    def __init__(self, hanger_template_path, connector_template_path):
        """
        Initialize detector: load templates, apply histogram equalization, compute SIFT/ORB features.
        
        Parameters:
        -----------
        hanger_template_path : str - Path to Hanger template image
        connector_template_path : str - Path to Connector template image
        """
        # Load templates in grayscale
        self.hanger_template = cv.imread(hanger_template_path, cv.IMREAD_GRAYSCALE)
        self.connector_template = cv.imread(connector_template_path, cv.IMREAD_GRAYSCALE)
        
        print(f"Hanger template shape: {self.hanger_template.shape}")
        print(f"Connector template shape: {self.connector_template.shape}")
        
        # Apply histogram equalization to normalize illumination
        self.hanger_template_eq = cv.equalizeHist(self.hanger_template)
        self.connector_template_eq = cv.equalizeHist(self.connector_template)
        
        # Initialize SIFT detector (contrastThreshold=0.03 for subtle features)
        self.sift = cv.SIFT_create(contrastThreshold=0.03, edgeThreshold=10)
        self.hanger_kp, self.hanger_des = self.sift.detectAndCompute(
            self.hanger_template, None
        )
        self.connector_kp, self.connector_des = self.sift.detectAndCompute(
            self.connector_template, None
        )
        
        # Initialize ORB as fallback for sparse SIFT features
        self.orb = cv.ORB_create(500)
        self.hanger_kp_orb, self.hanger_des_orb = self.orb.detectAndCompute(self.hanger_template, None)
        self.connector_kp_orb, self.connector_des_orb = self.orb.detectAndCompute(self.connector_template, None)
    
    
    def detect_objects_in_image(self, image_path, scales=None):
        """
        Main detection pipeline: 5-stage object detection and verification process.
        
        Stages: (1) Image preprocessing, (2) Multi-scale template matching,
                (3) Non-Maximum Suppression, (4) SIFT verification, (5) Result packaging
        
        Parameters:
        -----------
        image_path : str - Input image path
        scales : np.ndarray, optional - Scale factors (default: 25 scales from 0.3 to 1.5)
        
        Returns:
        --------
        dict with keys:
            'image': BGR color image
            'image_gray': Grayscale image
            'detections': {'hanger': [...], 'connector': [...]}
                Each detection: {'bbox': ((x1,y1),(x2,y2)), 'score': float, 
                                'scale': float, 'type': str, 'sift_matches': int}
        """
        
        # Load and preprocess image
        image_color = cv.imread(image_path)
        image = cv.cvtColor(image_color, cv.COLOR_BGR2GRAY)
        image_eq = cv.equalizeHist(image)
        
        print(f"\nInput image shape: {image.shape}")
        
        # Generate multi-scale search space
        if scales is None:
            scales = np.linspace(0.3, 1.5, 25)
        
        # Detect Hanger objects (match_threshold=0.45 for distinctive features)
        hanger_detections = self._detect_template(
            image_eq, self.hanger_template_eq, 
            self.hanger_template.shape,
            scales=scales, label='hanger', match_threshold=0.45
        )
        
        # Detect Connector objects (match_threshold=0.33 for subtle variations)
        connector_detections = self._detect_template(
            image_eq, self.connector_template_eq,
            self.connector_template.shape, 
            scales=scales, label='connector', match_threshold=0.33
        )
        
        print(f"Raw hanger detections: {len(hanger_detections)}")
        print(f"Raw connector detections: {len(connector_detections)}")
        
        # Apply Non-Maximum Suppression
        hanger_detections = self._apply_nms(hanger_detections, iou_threshold=0.5)
        connector_detections = self._apply_nms(connector_detections, iou_threshold=0.4)
        
        print(f"After NMS hanger: {len(hanger_detections)}")
        print(f"After NMS connector: {len(connector_detections)}")
        
        # SIFT feature verification
        image_sift_kp, image_sift_des = self.sift.detectAndCompute(image, None)
        
        if image_sift_des is not None:
            hanger_detections = self._verify_with_sift(
                hanger_detections, image, image_sift_kp, image_sift_des,
                self.hanger_des, label='hanger', min_matches=3
            )
            connector_detections = self._verify_with_sift(
                connector_detections, image, image_sift_kp, image_sift_des,
                self.connector_des, label='connector', min_matches=3
            )
        
        print(f"After SIFT verification hanger: {len(hanger_detections)}")
        print(f"After SIFT verification connector: {len(connector_detections)}")
        
        return {
            'image': image_color,
            'detections': {
                'hanger': hanger_detections,
                'connector': connector_detections
            },
            'image_gray': image
        }
    
    
    def _detect_template(self, image, template, template_shape, scales, label='', match_threshold=0.4):
        """
        Multi-scale template matching with local maximum extraction.
        
        Algorithm:
        1. For each scale: resize template, compute normalized correlation (TM_CCOEFF_NORMED)
        2. Extract local maxima using 3x3 morphological dilation
        3. Filter by match_threshold and image boundaries
        4. Return detections sorted by confidence score
        
        Parameters:
        -----------
        image : np.ndarray - Histogram-equalized input image
        template : np.ndarray - Grayscale template
        template_shape : tuple - Original template (height, width)
        scales : np.ndarray - Scale factors [0.3, 0.34, ..., 1.5]
        label : str - Object type ('hanger' or 'connector')
        match_threshold : float - Minimum correlation coefficient (0.33-0.45)
        
        Returns:
        --------
        list : Detections sorted by score descending
            [{'bbox': ((x1,y1),(x2,y2)), 'score': float, 'scale': float, 'type': str}, ...]
        
        Technical details:
        - TM_CCOEFF_NORMED: Normalized cross-correlation, robust to illumination
        - Local max extraction reduces clustering artifacts from template matching
        - Avoids brute-force threshold which generates many false positives
        """
        
        detections = []
        h, w = template_shape
        
        for scale in scales:
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            
            # Skip invalid sizes
            if scaled_w < 10 or scaled_h < 10:
                continue
            if scaled_w > image.shape[1] or scaled_h > image.shape[0]:
                continue
            
            # Resize template and perform matching
            scaled_template = cv.resize(template, (scaled_w, scaled_h))
            result = cv.matchTemplate(image, scaled_template, cv.TM_CCOEFF_NORMED)

            # Extract local maxima via morphological dilation
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv.dilate(result, kernel)
            local_max = (result == dilated) & (result >= match_threshold)
            locations = np.where(local_max)

            # Convert peak locations to bounding boxes
            for pt in zip(*locations[::-1]):
                score = float(result[pt[1], pt[0]])
                x1, y1 = int(pt[0]), int(pt[1])
                x2, y2 = x1 + scaled_w, y1 + scaled_h
                
                # Clamp to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                detections.append({
                    'bbox': ((x1, y1), (x2, y2)),
                    'score': float(score),
                    'scale': scale,
                    'type': label
                })
        
        # Sort by score (highest first)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        return detections
    
    
    def _apply_nms(self, detections, iou_threshold=0.3):
        """
        Non-Maximum Suppression (NMS) using greedy IoU-based filtering.
        
        Eliminates overlapping detections by iterating through sorted detections (highest score first)
        and removing any detection that overlaps with a kept detection beyond iou_threshold.
        
        Parameters:
        -----------
        detections : list - Detection list (sorted by score descending)
        iou_threshold : float - IoU threshold (Hanger: 0.5, Connector: 0.4)
        
        Returns:
        --------
        list : Filtered detections with overlaps removed
        
        Typical reduction: 20-50% fewer detections than input
        """
        
        if len(detections) == 0:
            return []
        
        final_detections = []
        
        for detection in detections:
            is_duplicate = False
            for final_det in final_detections:
                if self._calculate_iou(detection['bbox'], final_det['bbox']) > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        return final_detections
        
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        IoU = Intersection_Area / Union_Area
            = (x_overlap * y_overlap) / (area1 + area2 - intersection)
        
        Parameters:
        -----------
        box1, box2 : tuple - Bounding boxes in format ((x_min, y_min), (x_max, y_max))
        
        Returns:
        --------
        float : IoU value in range [0.0, 1.0]
            0.0: No overlap | 0.5: 50% intersection | 1.0: Identical boxes
        """
        
        (x1_min, y1_min), (x1_max, y1_max) = box1
        (x2_min, y2_min), (x2_max, y2_max) = box2
        
        # Calculate x and y overlaps
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate areas and union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    
    def _verify_with_sift(self, detections, image, image_kp, image_des, 
                         template_des, label='', min_matches=3):
        """
        Verify detections using SIFT feature matching with Lowe's ratio test.
        
        Two-tier approach:
        1. Primary: SIFT-based (FLANN matcher, Lowe's ratio test, min_matches=3)
        2. Fallback: ORB-based when sparse SIFT features (Hamming distance, min_matches=6)
        
        Both methods filter ambiguous matches using ratio test:
        - SIFT: distance_1 < 0.7 * distance_2
        - ORB: distance_1 < 0.75 * distance_2
        
        Parameters:
        -----------
        detections : list - Detection list with 'bbox' keys
        image : np.ndarray - Grayscale input image
        image_kp : list - SIFT keypoints from image
        image_des : np.ndarray - SIFT descriptors from image
        template_des : np.ndarray - SIFT descriptors from template
        label : str - 'hanger' or 'connector'
        min_matches : int - Minimum feature matches required (default=3)
        
        Returns:
        --------
        list : Verified detections (subset of input) sorted by score
               Each detection includes 'sift_matches' count
        """
        
        if template_des is None or image_des is None:
            return detections
        
        # Initialize FLANN matcher for SIFT (KD-tree based)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        
        verified = []
        
        for detection in detections:
            (x1, y1), (x2, y2) = detection['bbox']
            
            # Extract SIFT keypoints within detection ROI
            roi_kp_indices = [i for i, kp in enumerate(image_kp) 
                             if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2]
            
            # Fallback to ORB if sparse SIFT features
            if len(roi_kp_indices) < min_matches:
                roi = image[y1:y2, x1:x2]
                kp_roi_orb, des_roi_orb = self.orb.detectAndCompute(roi, None)
                if des_roi_orb is None:
                    continue

                template_des_orb = self.hanger_des_orb if label == 'hanger' else self.connector_des_orb
                if template_des_orb is None:
                    continue

                # ORB matching with Hamming distance
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
                try:
                    matches_orb = bf.knnMatch(template_des_orb, des_roi_orb, k=2)
                except Exception:
                    continue

                # Lowe's ratio test for ORB
                good_orb = [m for m_n in matches_orb if len(m_n) == 2 
                           for m, n in [m_n] if m.distance < 0.75 * n.distance]

                if len(good_orb) >= max(min_matches, 6):
                    detection['sift_matches'] = len(good_orb)
                    verified.append(detection)
                continue

            roi_des = image_des[roi_kp_indices]
            
            # SIFT matching with FLANN
            try:
                matches = flann.knnMatch(template_des, roi_des, k=2)
            except:
                continue
            
            # Lowe's ratio test for SIFT
            good = [m for match_pair in matches if len(match_pair) == 2 
                   for m, n in [match_pair] if m.distance < 0.7 * n.distance]
            
            if len(good) >= min_matches:
                detection['sift_matches'] = len(good)
                verified.append(detection)
        
        # Sort by score
        verified = sorted(verified, key=lambda x: x.get('score', 0), reverse=True)
        return verified
    
    
    def visualize_detections(self, result_data, save_path=None):
        """
        Generate 2x2 visualization with: (top-left) annotated image, (top-right) statistics,
        (bottom-left) Hanger template, (bottom-right) Connector template.
        
        Hanger detections: Green rectangles | Connector detections: Red rectangles
        Each label includes confidence score and SIFT match count [M:n]
        
        Parameters:
        -----------
        result_data : dict - Dictionary from detect_objects_in_image()
        save_path : str, optional - Save figure to file at 150 DPI
        
        Returns:
        --------
        np.ndarray : Annotated BGR image with bounding boxes
        """
        
        image_color = result_data['image'].copy()
        detections = result_data['detections']
        
        hanger_color = (0, 255, 0)      # Green
        connector_color = (0, 0, 255)   # Red
        
        # Draw Hanger detections
        for i, detection in enumerate(detections['hanger']):
            (x1, y1), (x2, y2) = detection['bbox']
            cv.rectangle(image_color, (x1, y1), (x2, y2), hanger_color, 2)
            
            label = f"Hanger#{i+1} ({detection['score']:.3f})"
            if 'sift_matches' in detection:
                label += f" [M:{detection['sift_matches']}]"
            
            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv.rectangle(image_color, (x1, y1 - text_size[1] - 5),
                        (x1 + text_size[0], y1), hanger_color, -1)
            cv.putText(image_color, label, (x1, y1 - 5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw Connector detections
        for i, detection in enumerate(detections['connector']):
            (x1, y1), (x2, y2) = detection['bbox']
            cv.rectangle(image_color, (x1, y1), (x2, y2), connector_color, 2)
            
            label = f"Connector#{i+1} ({detection['score']:.3f})"
            if 'sift_matches' in detection:
                label += f" [M:{detection['sift_matches']}]"
            
            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv.rectangle(image_color, (x1, y1 - text_size[1] - 5),
                        (x1 + text_size[0], y1), connector_color, -1)
            cv.putText(image_color, label, (x1, y1 - 5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot [0,0]: Detection results
        axes[0, 0].imshow(cv.cvtColor(image_color, cv.COLOR_BGR2RGB))
        axes[0, 0].set_title('Detection Results (Hanger: Green, Connector: Red)', 
                             fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Subplot [0,1]: Statistics
        axes[0, 1].axis('off')
        stats_text = f"""
        === Detection Statistics ===
        
        Hangers detected: {len(detections['hanger'])}
        Connectors detected: {len(detections['connector'])}
        Total objects: {len(detections['hanger']) + len(detections['connector'])}
        
        === Hanger Detections ===
        """
        for i, det in enumerate(detections['hanger']):
            stats_text += f"\n#{i+1}: Score={det['score']:.3f}"
            if 'sift_matches' in det:
                stats_text += f", SIFT Matches={det['sift_matches']}"
        
        stats_text += "\n\n=== Connector Detections ===\n"
        for i, det in enumerate(detections['connector']):
            stats_text += f"\n#{i+1}: Score={det['score']:.3f}"
            if 'sift_matches' in det:
                stats_text += f", SIFT Matches={det['sift_matches']}"
        
        axes[0, 1].text(0.05, 0.95, stats_text, transform=axes[0, 1].transAxes,
                       fontfamily='monospace', fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Subplot [1,0]: Hanger template
        axes[1, 0].imshow(self.hanger_template, cmap='gray')
        axes[1, 0].set_title('Baseline_hanger.png', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Subplot [1,1]: Connector template
        axes[1, 1].imshow(self.connector_template, cmap='gray')
        axes[1, 1].set_title('Baseline_connector.png', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Result saved to: {save_path}")
        
        plt.show()
        return image_color
    
    
    def print_detection_summary(self, result_data):
        """
        Print formatted console output of detection results.
        
        Displays per-object statistics including confidence scores, positions, dimensions, 
        and SIFT verification counts (if available).
        """
        
        detections = result_data['detections']
        
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        
        print(f"\n【 HANGER 】 Total: {len(detections['hanger'])} detected")
        print("-" * 60)
        for i, det in enumerate(detections['hanger'], 1):
            (x1, y1), (x2, y2) = det['bbox']
            w = x2 - x1
            h = y2 - y1
            print(f"  #{i} | Score: {det['score']:.4f} | "
                  f"Position: ({x1},{y1}) | Size: {w}x{h}", end="")
            if 'sift_matches' in det:
                print(f" | SIFT Matches: {det['sift_matches']}", end="")
            print()
        
        print(f"\n【 CONNECTOR 】 Total: {len(detections['connector'])} detected")
        print("-" * 60)
        for i, det in enumerate(detections['connector'], 1):
            (x1, y1), (x2, y2) = det['bbox']
            w = x2 - x1
            h = y2 - y1
            print(f"  #{i} | Score: {det['score']:.4f} | "
                  f"Position: ({x1},{y1}) | Size: {w}x{h}", end="")
            if 'sift_matches' in det:
                print(f" | SIFT Matches: {det['sift_matches']}", end="")
            print()
        
        print("\n" + "="*60 + "\n")


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================

if __name__ == '__main__':
    """
    Main execution workflow demonstrating the complete detection pipeline.
    
    Input files: Baseline_hanger.png, Baseline_connector.png, vision_sensor_0.png
    Output file: detection_results.png
    """
    
    print("=" * 70)
    print("Hanger & Connector Differentiation Algorithm")
    print("=" * 70)
    
    # Initialize detector with template images
    detector = HangerConnectorDetector(
        'Baseline_hanger.png',
        'Baseline_connector.png'
    )
    
    # Run detection pipeline on input image
    result = detector.detect_objects_in_image('vision_sensor_0.png')
    
    # Output console summary
    detector.print_detection_summary(result)
    
    # Generate and save visualization
    detector.visualize_detections(result, save_path='detection_results.png')
