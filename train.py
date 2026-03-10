from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.train(
    data="Vietnamese Food.v1i.yolov8/data.yaml",
    epochs=30,
    imgsz=640,
    project=".",
    name="result",
    exist_ok=True
)