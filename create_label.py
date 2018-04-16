'''
用于建立label.csv这个文件
'''
import csv
import os
with open("label.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for path, directories, filenames in os.walk("data/train"):
        for filename in filenames:
            if "dog" in filename:
                label = 1
            else:
                label = 0
            writer.writerow([filename, label])

