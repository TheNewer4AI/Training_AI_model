import os
import numpy as np
import shutil

# Link to image folder
normal_folder = '/root/data/fruit/apple/Preprocessing_3_U2Net/Normal_cropped/'
abnormal_folder = '/root/data/fruit/apple/Preprocessing_3_U2Net/Abnormal_cropped/'
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

# list image files
abnormal_files = os.listdir(abnormal_folder)
normal_files = os.listdir(normal_folder)

print("NG: ", len(abnormal_files))
print("OK: ", len(normal_files))

# mixing name
np.random.shuffle(abnormal_files)
np.random.shuffle(normal_files)

# split data
abnormal_train_split = int(train_ratio * len(abnormal_files))
abnormal_valid_split = int((train_ratio + valid_ratio) * len(abnormal_files))
print("abnormal_train_split: ", abnormal_train_split)
print("abnormal_valid_split: ", abnormal_valid_split)

normal_train_split = int(train_ratio * len(normal_files))
normal_valid_split = int((train_ratio + valid_ratio) * len(normal_files))
print("normal_train_split: ", normal_train_split)
print("normal_valid_split: ", normal_valid_split)

#split name
abnormal_train_files = abnormal_files[:abnormal_train_split]
abnormal_valid_files = abnormal_files[abnormal_train_split:abnormal_valid_split]
abnormal_test_files = abnormal_files[abnormal_valid_split:]

normal_train_files = normal_files[:normal_train_split]
normal_valid_files = normal_files[normal_train_split:normal_valid_split]
normal_test_files = normal_files[normal_valid_split:]

# Creating Train / Val / Test folders (One time use)

src_ng = '/root/data/fruit/apple/Preprocessing_3_U2Net/Abnormal_cropped/'
src_ok = '/root/data/fruit/apple/Preprocessing_3_U2Net/Normal_cropped/'

target_train_ng = '/root/data/fruit/apple/Preprocessing_3_U2Net/train/NG/'
target_train_ok = '/root/data/fruit/apple/Preprocessing_3_U2Net/train/OK/'

target_val_ng = '/root/data/fruit/apple/Preprocessing_3_U2Net/val/NG/'
target_val_ok = '/root/data/fruit/apple/Preprocessing_3_U2Net/val/OK/'

target_test_ng = '/root/data/fruit/apple/Preprocessing_3_U2Net/test/NG/'
target_test_ok = '/root/data/fruit/apple/Preprocessing_3_U2Net/test/OK/'

# copy images
#NG cases
print("Copying train NG images")
for name_train in abnormal_train_files:
  # print("name_train: ", name_train)
  sour_dir = src_ng + name_train
  shutil.copy(sour_dir, target_train_ng)

print("Copying validation NG images")
for name_val in abnormal_valid_files:
  # print("name_val: ", name_val)
  sour_dir = src_ng + name_val
  shutil.copy(sour_dir, target_val_ng)

print("Copying test NG images")
for name_test in abnormal_test_files:
  # print("name_test: ", name_test)
  sour_dir = src_ng + name_test
  shutil.copy(sour_dir, target_test_ng)

#OK cases
print("Copying train OK images")
for name_train in normal_train_files:
  # print("name_train: ", name_train)
  sour_dir = src_ok + name_train
  shutil.copy(sour_dir, target_train_ok)

print("Copying validation OK images")
for name_test in normal_valid_files:
  # print("name_test: ", name_test)
  sour_dir = src_ok + name_test
  shutil.copy(sour_dir, target_val_ok)

print("Copying test OK images")
for name_val in normal_test_files:
  # print("name_val: ", name_val)
  sour_dir = src_ok + name_val
  shutil.copy(sour_dir, target_test_ok)

print("finish.............")





