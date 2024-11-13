import os
import shutil
import random

source_dir = 'data'
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

random.shuffle(all_files)

train_end = int(len(all_files) * train_ratio)
val_end = train_end + int(len(all_files) * val_ratio)

train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

for file in train_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, file))

for file in test_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))

print(f"Moved {len(train_files)} files to {train_dir}")
print(f"Moved {len(val_files)} files to {val_dir}")
print(f"Moved {len(test_files)} files to {test_dir}")
