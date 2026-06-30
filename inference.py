import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os
from datetime import datetime
import time

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL", "0")
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 640))
FIRE_SMOKE_FRAME_THRESHOLD = int(os.getenv("FIRE_SMOKE_FRAME_THRESHOLD", 30))
DANGER_ZONE_ENABLED = os.getenv("DANGER_ZONE_ENABLED", "true").lower() == "true"
AFTER_HOURS_START = os.getenv("AFTER_HOURS_START", "18:00")
AFTER_HOURS_END = os.getenv("AFTER_HOURS_END", "06:00")
ALERT_LOG_PATH = os.getenv("ALERT_LOG_PATH", "logs/alerts.txt")

CLASS_NAMES = [
    "helmet", "safety_boots", "safety_vest",
    "fire", "smoke",
    "person_authorized", "person_unknown",
    "car", "truck", "bike", "animal"
]

CLASS_COLORS = {
    "helmet": (0, 255, 0),
    "safety_boots": (0, 200, 100),
    "safety_vest": (0, 150, 255),
    "fire": (0, 0, 255),
    "smoke": (100, 100, 100),
    "person_authorized": (255, 255, 0),
    "person_unknown": (0, 0, 200),
    "car": (255, 165, 0),
    "truck": (255, 0, 0),
    "bike": (0, 255, 255),
    "animal": (128, 0, 128)
}

class SafetySystem:
    def __init__(self):
        # Check if custom model exists, otherwise use default
        if os.path.exists(MODEL_PATH):
            print(f"✅ Loading custom model: {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
            self.use_custom_classes = True
        else:
            print(f"⚠️ Custom model not found at {MODEL_PATH}")
            print(f"📥 Loading default YOLO11n model (for demonstration)")
            self.model = YOLO("yolo11n.pt")
            self.use_custom_classes = False
        
        # Get class names from the model
        self.class_names = self.model.names
        print(f"📋 Loaded {len(self.class_names)} classes")
        
        self.fire_smoke_counter = 0
        self.alerts = []
        self.danger_zone = []
        self.drawing_zone = False
        self.current_point = None

    def is_after_hours(self):
        now = datetime.now().time()
        start = datetime.strptime(AFTER_HOURS_START, "%H:%M").time()
        end = datetime.strptime(AFTER_HOURS_END, "%H:%M").time()
        if start <= end:
            return now >= start and now <= end
        else:
            return now >= start or now <= end

    def point_in_polygon(self, point, polygon):
        if len(polygon) < 3:
            return False
        x, y = point
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def box_in_zone(self, box, polygon):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return self.point_in_polygon((center_x, center_y), polygon)

    def check_ppe_compliance(self, results):
        if not self.use_custom_classes:
            return []  # Skip PPE check for default model
            
        non_compliant = []
        persons = []
        helmets = []
        vests = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                if class_name in ["person_authorized", "person_unknown"]:
                    persons.append({"bbox": (x1, y1, x2, y2), "class": class_name, "conf": conf})
                elif class_name == "helmet":
                    helmets.append({"bbox": (x1, y1, x2, y2), "class": class_name, "conf": conf})
                elif class_name == "safety_vest":
                    vests.append({"bbox": (x1, y1, x2, y2), "class": class_name, "conf": conf})

        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            has_helmet = False
            has_vest = False

            for helmet in helmets:
                hx1, hy1, hx2, hy2 = helmet["bbox"]
                if hx1 >= px1 and hy1 >= py1 and hx2 <= px2 and hy2 <= py2:
                    has_helmet = True
                    break

            for vest in vests:
                vx1, vy1, vx2, vy2 = vest["bbox"]
                if vx1 >= px1 and vy1 >= py1 and vx2 <= px2 and vy2 <= py2:
                    has_vest = True
                    break

            if not has_helmet or not has_vest:
                non_compliant.append({
                    "person": person,
                    "missing_helmet": not has_helmet,
                    "missing_vest": not has_vest
                })

        return non_compliant

    def check_fire_smoke_persistence(self, results):
        if not self.use_custom_classes:
            return False  # Skip fire/smoke check for default model
            
        has_fire_smoke = False
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                if class_name in ["fire", "smoke"]:
                    has_fire_smoke = True
                    break

        if has_fire_smoke:
            self.fire_smoke_counter += 1
        else:
            self.fire_smoke_counter = 0

        return self.fire_smoke_counter >= FIRE_SMOKE_FRAME_THRESHOLD

    def check_intrusion(self, results):
        intruders = []
        if not DANGER_ZONE_ENABLED or len(self.danger_zone) < 3:
            return intruders

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                target_classes = ["person_unknown", "animal", "car", "truck", "bike"] if self.use_custom_classes else ["person", "car", "truck", "motorcycle", "bicycle", "dog", "cat"]
                
                if class_name in target_classes:
                    if self.box_in_zone((x1, y1, x2, y2), self.danger_zone):
                        intruders.append({"class": class_name, "bbox": (x1, y1, x2, y2)})

        return intruders

    def log_alert(self, alert_type, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{alert_type}] {message}"
        print(log_entry)
        self.alerts.append(log_entry)
        with open(ALERT_LOG_PATH, "a") as f:
            f.write(log_entry + "\n")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_zone = True
            self.danger_zone.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_zone:
            self.current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_zone = False
            self.current_point = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.danger_zone = []
            self.current_point = None

    def run(self):
        # Try to open video stream
        if RTSP_URL == "0":
            print("Trying to open webcam...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows
            
            # If that fails, try without CAP_DSHOW
            if not cap.isOpened():
                print("DirectShow failed, trying default method...")
                cap = cv2.VideoCapture(0)
        else:
            print(f"Trying to open RTSP stream: {RTSP_URL}")
            cap = cv2.VideoCapture(RTSP_URL)

        if not cap.isOpened():
            print("Error: Could not open video stream!")
            print("\nTroubleshooting tips:")
            print("1. For webcam: Make sure no other app is using it")
            print("2. For RTSP: Check URL, username, password, and camera connectivity")
            print("3. Check your .env file to verify RTSP_URL is set correctly")
            return

        print("Video stream opened successfully!")
        cv2.namedWindow("Construction Site Safety Monitor")
        cv2.setMouseCallback("Construction Site Safety Monitor", self.mouse_callback)

        fps_start_time = time.time()
        fps_counter = 0
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            fps_counter += 1
            if time.time() - fps_start_time >= 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMAGE_SIZE)

            annotated_frame = frame.copy()

            if len(self.danger_zone) >= 3:
                cv2.polylines(annotated_frame, [np.array(self.danger_zone, np.int32)], True, (0, 0, 255), 2)
            if self.drawing_zone and len(self.danger_zone) > 0:
                temp_zone = self.danger_zone + [self.current_point]
                cv2.polylines(annotated_frame, [np.array(temp_zone, np.int32)], True, (0, 255, 255), 2)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.class_names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])

                    color = CLASS_COLORS.get(class_name, (0, 255, 255))  # Default cyan if not found
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            non_compliant = self.check_ppe_compliance(results)
            for nc in non_compliant:
                px1, py1, px2, py2 = nc["person"]["bbox"]
                cv2.rectangle(annotated_frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 3)
                alert_msg = []
                if nc["missing_helmet"]:
                    alert_msg.append("NO HELMET")
                if nc["missing_vest"]:
                    alert_msg.append("NO VEST")
                alert_label = " | ".join(alert_msg)
                cv2.putText(annotated_frame, alert_label, (int(px1), int(py1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.log_alert("PPE NON-COMPLIANCE", f"Person missing {alert_label}")

            fire_smoke_alert = self.check_fire_smoke_persistence(results)
            if fire_smoke_alert:
                cv2.putText(annotated_frame, "FIRE/SMOKE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                self.log_alert("FIRE/SMOKE", "Fire/Smoke detected for 30+ frames")

            if self.is_after_hours():
                intruders = self.check_intrusion(results)
                for intruder in intruders:
                    ix1, iy1, ix2, iy2 = intruder["bbox"]
                    cv2.rectangle(annotated_frame, (int(ix1), int(iy1)), (int(ix2), int(iy2)), (0, 0, 200), 3)
                    self.log_alert("INTRUSION", f"{intruder['class']} entered danger zone after hours")

            cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Construction Site Safety Monitor", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.danger_zone = []
                self.current_point = None

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SafetySystem()
    system.run()
