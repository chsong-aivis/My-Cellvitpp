import csv

def load_filename_map(map_csv):
    orig2new, new2orig = {}, {}
    with open(map_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig2new[row['original_name']] = row['new_name']
            new2orig[row['new_name']] = row['original_name']
    return orig2new, new2orig

if __name__ == "__main__":
    orig2new, new2orig = load_filename_map("./Dataset_Old/filename_map.csv")
    # 예시: 원본 → 새 파일명
    print("원본 예시:", list(orig2new.items())[:5])
    # 예시: 새 파일명 → 원본
    print("새 파일명 예시:", list(new2orig.items())[:5]) 