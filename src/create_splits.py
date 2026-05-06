import os
from sklearn.model_selection import train_test_split

IMG_DIR = "data/processed/images"
SPLIT_DIR = "data/splits"

os.makedirs(SPLIT_DIR, exist_ok=True)

files = [f.split(".")[0] for f in os.listdir(IMG_DIR)]

train, temp = train_test_split(files, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)


def save_split(name, data):
    with open(os.path.join(SPLIT_DIR, f"{name}.txt"), "w") as f:
        for item in data:
            f.write(item + "\n")


save_split("train", train)
save_split("val", val)
save_split("test", test)

print("Splits created ✅")
