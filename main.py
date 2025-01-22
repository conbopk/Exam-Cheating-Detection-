import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf


dataset_paths = {
    "cheating": "ExamCheatingDataset/train/cheating",
    "giving_code": "ExamCheatingDataset/train/giving code",
    "giving_object": "ExamCheatingDataset/train/giving object",
    "looking_friend": "ExamCheatingDataset/train/looking friend",
    "normal_act": "ExamCheatingDataset/train/normal act"
}

image_paths = []
labels = []

for label, path in dataset_paths.items():

    if os.path.exists(path):
        files = os.listdir(path)
        for file in files:
            image_paths.append(os.path.join(path, file))
            labels.append(label)

    else:
        print(f"Path does not exist: {path}")

df = pd.DataFrame({"image_path": image_paths, "label": labels})
# print(df.shape)
# print(df.columns)
# print(df.info)
# print(df.duplicated().sum())
# print(df.isnull().sum())
# print(df.nunique())
# print(df['label'].unique())
# print(df['label'].value_counts())


def display_images_from_categories(dataset_paths, num_images=5):
    plt.figure(figsize=(15,15))
    image_idx = 1

    for label, path in dataset_paths.items():
        if os.path.exists(path):
            files = os.listdir(path)[:num_images]
            for file in files:
                file_path = os.path.join(path, file)
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.subplot(len(dataset_paths), num_images, image_idx)
                plt.imshow(image)
                plt.axis('off')
                plt.title(label, fontsize=10)
                image_idx+=1
        else:
            print(f"Path does not exist: {path}")

    plt.tight_layout()
    plt.show()

# display_images_from_categories(dataset_paths, num_images=5)

# sns.set_theme(style='whitegrid')
# plt.figure(figsize=(10,6))
# sns.countplot(data=df, x='label', order=df['label'].value_counts().index, palette='viridis', hue='label', legend=False)
# plt.title("Distribution of Labels (Countplot)", fontsize=16)
# plt.xlabel("Type", fontsize=14)
# plt.ylabel("Count", fontsize=14)
# plt.xticks(rotation=45, fontsize=12)
# plt.show()
#
#
# plt.figure(figsize=(8,8))
# df['label'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(df['label'].unique())))
# plt.title("Distribution of Labels (Pie Chart)", fontsize=16)
# plt.show()



