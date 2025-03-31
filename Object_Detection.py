import cv2
from ultralytics import YOLO

# Load YOLOv8 Model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # You can also use "yolov8s.pt" for better accuracy

# Open Webcam (0 = default webcam, change if using an external camera)
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()  # Read a frame from the webcam
    
    if not ret:
        break

    # Run YOLO object detection on the frame
    results = model(frame)

    # Draw detections on the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"  # Object label

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Show the video feed with detections
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV window
video.release()
cv2.destroyAllWindows()
