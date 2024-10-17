from ultralytics import YOLO
import cv2

def display_boundary_box():
    # Load the model
    model = YOLO('yolov8n.pt')

    # Run detection on the image
    results = model('val image directory')

    # Get image with bounding boxes
    image = results[0].plot()

    # Display the image
    cv2.imshow('YOLOv8 Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def checking_correct_dataset():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Load dataset from dataset.yaml
    data = 'dataset.yaml'

    # Use the train2 mode and pass the dataset to the model
    results = model.train(data=data, epochs=50, imgsz=1056)


if __name__ == "__main__":
    # Define paths and initialize model
    model = YOLO('yolov8n.pt')
    model.train(data='best.pt or last.pt file directory', epochs=50, imgsz=1056)
