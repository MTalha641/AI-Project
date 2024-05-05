import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.saved_model.load(r'C:\Users\DELL\Desktop\Semester 6\AI Project\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model')

# Load label map from pbtxt file
def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'id:' in line:
                id_index = line.index('id:') + 3
                id_value = int(line[id_index:].strip())
            elif 'name:' in line:
                name_index = line.index('name:') + 5
                name_value = line[name_index:].strip().strip('"')
                label_map[id_value] = {'id': id_value, 'name': name_value}
    return label_map

# Load label map from pbtxt file
# Update the path to your label map file
label_map = load_label_map(r'C:\Users\DELL\Desktop\Semester 6\AI Project\label_map.pbtxt')

# Adjusted minimum confidence threshold
min_confidence = 0.7  # Adjust this value as needed

# Perform real-time object detection
def main():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

    
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform inference
        detections = model(input_tensor)

        
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        for i in range(len(boxes)):
            if scores[i] > min_confidence:  
                ymin, xmin, ymax, xmax = boxes[i]
                left = int(xmin * frame.shape[1])
                top = int(ymin * frame.shape[0])
                right = int(xmax * frame.shape[1])
                bottom = int(ymax * frame.shape[0])
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                class_id = int(classes[i])  # Convert class to integer
                class_name = label_map[class_id]['name'] if class_id in label_map else 'unknown'  # Get class name from label_map
                cv2.putText(frame, class_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Real-time Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
