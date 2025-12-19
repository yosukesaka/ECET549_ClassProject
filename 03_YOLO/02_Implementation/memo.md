3. 学習コマンド
yolo detect train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640


※ 20枚しかないため、small model (yolov8n) を推奨
※ epochs=50〜100 で十分

④ テスト画像で推論

vision_Sensor_0.png を検出したいとき：

yolo detect predict model=runs/detect/train/weights/best.pt source=vision_Sensor_0.png


出力は
runs/detect/predict/vision_Sensor_0.jpg
として保存されます。





yolo detect predict model=runs/detect/train/weights/best.pt source=images/test/vision_Sensor_0.png conf=0.1 show=True save=True

続けて学習
yolo detect train model=runs/detect/train2/weights/best.pt data=dataset.yaml epochs=300 imgsz=640   # 例. 全体で50エポックまで

最初から学習し直す場合
yolo detect train \
    model=yolov8n.pt \
    data=dataset.yaml \
    epochs=150 \
    imgsz=640

Augment
yolo detect train \
    model=yolov8n.pt \
    data=dataset.yaml \
    epochs=150 \
    imgsz=640 \
    augment=True

推論確認コマンド
yolo detect predict model=runs/detect/train/weights/best.pt source=images/test/vision_sensor_0.png conf=0.25 show=True save=True

yolo detect predict model=runs/detect/train/weights/best.pt source=images/test/ conf=0.25 show=True save=True

yolo detect predict model=runs/detect/train/weights/best.pt source="images/test/*.png" conf=0.25 show=True save=True

