from ultralytics import YOLO

model = YOLO("..//runs/detect//train7//weights//best.pt")

model.predict(source = "prova1.mp4", show = True, conf = 0.5, line_width = 1, save_crop = True)