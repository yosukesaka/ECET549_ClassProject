import os
import shutil
import random

# 元画像が入っているフォルダ
img_dir = "dataset_4/images/original"
lbl_dir = "dataset_4/labels/original"

# 出力先フォルダ
train_img_dir = "dataset_4/images/train"
val_img_dir = "dataset_4/images/val"
train_lbl_dir = "dataset_4/labels/train"
val_lbl_dir = "dataset_4/labels/val"

# 出力フォルダ作成
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# クラスID
CONNECTOR_ID = "0"
HANGER_ID = "1"

# クラスごとの画像リスト
connector_imgs = []
hanger_imgs = []

# 画像一覧
images = [f for f in os.listdir(img_dir) if f.endswith(('.png','.jpg','.jpeg'))]

# ラベルに基づいて分類
for img in images:
    label = img.rsplit('.',1)[0] + ".txt"
    label_path = os.path.join(lbl_dir, label)

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        first_line = f.readline().strip()
        class_id = first_line.split()[0]

        if class_id == CONNECTOR_ID:
            connector_imgs.append(img)
        elif class_id == HANGER_ID:
            hanger_imgs.append(img)

# ランダムシャッフル
random.shuffle(connector_imgs)
random.shuffle(hanger_imgs)

def split_and_copy(file_list, train_ratio=0.8):
    split_idx = int(len(file_list) * train_ratio)
    return file_list[:split_idx], file_list[split_idx:]

# 8:2 に分割
connector_train, connector_val = split_and_copy(connector_imgs)
hanger_train, hanger_val = split_and_copy(hanger_imgs)

# ファイルコピー関数
def copy_files(file_list, img_dst, lbl_dst):
    for img in file_list:
        label = img.rsplit('.',1)[0] + ".txt"
        shutil.copy(os.path.join(img_dir, img), img_dst)
        shutil.copy(os.path.join(lbl_dir, label), lbl_dst)

# コピー実行
copy_files(connector_train, train_img_dir, train_lbl_dir)
copy_files(connector_val, val_img_dir, val_lbl_dir)
copy_files(hanger_train, train_img_dir, train_lbl_dir)
copy_files(hanger_val, val_img_dir, val_lbl_dir)

print("完了：connector/hanger を 8:2 でランダムに分割しました！")