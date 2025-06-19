import os
from pathlib import Path

def check_and_fix_split_file(split_file, images_dir, labels_dir, output_file):
    # 실제 파일명(stem) 목록
    image_stems = set(f.stem for f in Path(images_dir).glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"])
    label_stems = set(f.stem for f in Path(labels_dir).glob("*.csv"))
    # split 파일명(stem) 목록
    with open(split_file, "r") as f:
        split_stems = [line.strip() for line in f if line.strip()]
    # 실제 존재하는 파일만 필터링
    valid_stems = [s for s in split_stems if s in image_stems and s in label_stems]
    missing = [s for s in split_stems if s not in image_stems or s not in label_stems]
    print(f"{split_file} - 총 {len(split_stems)}개, 실제 존재 {len(valid_stems)}개, 누락 {len(missing)}개")
    if missing:
        print("누락/불일치 파일명 예시:", missing[:10])
    # 정정된 split 파일 저장
    with open(output_file, "w") as f:
        f.write("\n".join(valid_stems))
    print(f"정정된 split 파일 저장: {output_file}")

if __name__ == "__main__":
    base = "./Dataset_Old"
    check_and_fix_split_file(f"{base}/splits/fold_0/train.csv", f"{base}/train/images", f"{base}/train/labels", f"{base}/splits/fold_0/train_fixed.csv")
    check_and_fix_split_file(f"{base}/splits/fold_0/val.csv", f"{base}/train/images", f"{base}/train/labels", f"{base}/splits/fold_0/val_fixed.csv") 