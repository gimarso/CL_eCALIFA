import tensorflow as tf
import os
import re
import argparse

def extract_numbers(filename):
    # Extract the numbers from the filename
    numbers = re.findall(r'\d+', filename)
    # return a tuple of the second and third number if they exist, otherwise return 0
    return (int(numbers[1]), int(numbers[2])) if numbers else (0, 0)


# Set up argument parsing
parser = argparse.ArgumentParser(description='Merge tfrecord files')
parser.add_argument('--folder_path', required=True, help='Path to folder with tfrecord files')
parser.add_argument('--merged_folder_path', required=True, help='Path to folder for merged files')
args = parser.parse_args()

# list all tfrecord files and sort them
tfrecord_files = sorted([f for f in os.listdir(args.folder_path) if f.endswith('.tfrecord')])

# group files by the last number
groups = {}
for file_name in tfrecord_files:
    split_name = file_name.split('_')
    group_name = split_name[-1].split('.')[0]  # get the last number
    prefix_name = split_name[0]  # get the first part of the file name
    if group_name not in groups:
        groups[group_name] = {'files': [], 'prefix': prefix_name}
    groups[group_name]['files'].append(os.path.join(args.folder_path, file_name))

for v, value in enumerate(groups.keys()) :
    groups[value]['files'] = sorted(groups[value]['files'], key=extract_numbers)


# merge files in each group
for group_name, file_info in groups.items():
    merged_file_name = os.path.join(args.merged_folder_path, f'{file_info["prefix"]}_{group_name}.tfrecord')
    with tf.io.TFRecordWriter(merged_file_name) as writer:
        for file_name in file_info['files']:
            for record in tf.data.TFRecordDataset(file_name):
                writer.write(record.numpy())

