import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split

from util import constants as C

EXAMPLE_THRESHOLD = 10

def dataset_split(dataset_path, output_folder, train_prop, val_prop, test_prop):
    
    # Check if the splits add up to 1
    total = train_prop + val_prop + test_prop
    if total != 1:
        print("Train, validation, and test proportions must add up to 1, they are", total)

    data = pd.read_csv(dataset_path, index_col=0)
    data = data[data['Image_Folder'].apply(str) != 'nan']
    
    valid = False
    while not valid:
        # Split into train and val+test datasets
        train, val_test = train_test_split(data, test_size=val_prop+test_prop, random_state=0)

        # Split the val+test datasets into validation and test
        val, test = train_test_split(val_test, test_size=test_prop/(val_prop+test_prop), random_state=0)
        
        valid = True
        for label in C.class_labels_list:
            train_count = train['Type'].apply(lambda x : label in x).sum()
            val_count = val['Type'].apply(lambda x : label in x).sum()
            test_count = test['Type'].apply(lambda x : label in x).sum()
            if train_count < EXAMPLE_THRESHOLD or val_count < EXAMPLE_THRESHOLD or test_count < EXAMPLE_THRESHOLD:
                valid = False
                print (label, train_count, val_count, test_count)
    
    # Output train, val, test datasets
    train.to_csv(output_folder + "train_dataset.csv", index=False)
    val.to_csv(output_folder + "val_dataset.csv", index=False)
    test.to_csv(output_folder + "test_dataset.csv", index=False)

if __name__ == '__main__':
    # usage: python download.py [input data file] [output folder] [train percent] [val percent] [test percent]
    dataset_path = sys.argv[1]
    output_folder = sys.argv[2]
    train_prop = float(sys.argv[3]) / 100
    val_prop = float(sys.argv[4]) / 100
    test_prop = float(sys.argv[5]) / 100
    
    dataset_split(dataset_path, output_folder, train_prop, val_prop, test_prop)