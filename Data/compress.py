import sys
import os
import tarfile

current_dir = os.path.dirname(os.path.abspath(__file__))

commend = sys.argv[1]

if commend == '-c':

    target_file_list = [e for e in os.listdir(current_dir) if e.endswith('.pkl')]

    for file in target_file_list:

        print('Compressing', file)

        tar = tarfile.open(file.replace('.pkl', '.tar.gz'), "w:gz")

        tar.add(file)

        tar.close()

if commend == '-x':

    target_file_list = [e for e in os.listdir(current_dir) if e.endswith('.tar.gz')]

    for file in target_file_list:

        print('UnCompressing', file)

        tar = tarfile.open(file, "r:gz")

        file_names = tar.getnames()

        for file_name in file_names:
            tar.extract(file_name, current_dir)

        tar.close()
