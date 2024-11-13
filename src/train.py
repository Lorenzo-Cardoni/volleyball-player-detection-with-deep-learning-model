from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data = "../dataset//data.yaml", imgsz = 1216, 
            batch = 2, epochs = 100, workers = 0, device = 0)