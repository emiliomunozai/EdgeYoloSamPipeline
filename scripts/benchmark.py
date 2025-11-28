from ultralytics import YOLO
import time

def benchmark(model_path, name):
    model = YOLO(model_path)
    model.overrides["verbose"] = False

    print(f"\n{name}")

    # Warmup
    model.predict(
        source="datasets/coco/images/val2017",
        imgsz=640,
        device=0,
        save=False,
        show=False,
        verbose=False
    )

    t0 = time.time()

    metrics = model.val(
        data="coco.yaml",
        imgsz=640,
        device=0,
        verbose=False
    )

    t1 = time.time()

    print(f"Accuracy (mAP50-95): {metrics.box.map:.4f}")
    print(f"Total Time (s): {t1 - t0:.2f}")

if __name__ == "__main__":
    benchmark("models/final/yolov8n.pt", "FP32 model")
    benchmark("models/final/fp16_yolov8n.engine", "FP16 engine")
    benchmark("models/final/int8_yolov8n.engine", "INT8 engine")
