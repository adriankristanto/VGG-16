# reference: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb
# dataset preparation process follows the steps used in the reference
import pandas as pd 
import os 

# file containing the attributes of each image, such as the gender of the person in the image
# which will be used here
ATTR_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/list_attr_celeba.txt'
# file containing the recommended way of partitioning the data into train, validation and test dataset
# where 0=train, 1=validation and 2=test
EVAL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/list_eval_partition.txt'

# read the attr file
# it is separated with whitespaces, thus, we use \s+ which represents 1 or more whitespaces
# skiprows=1 means we skip 1 row from the top of the file, this is because the first row states the number of images
# usecols=['Male'] for gender classfication
attr_df = pd.read_csv(ATTR_PATH, sep='\s+', skiprows=1, usecols=['Male'])
# replace -1 with 0
attr_df.loc[lambda df: df['Male'] == -1] = 0
attr_df.columns = ['Label']

# read partition file
eval_df = pd.read_csv(EVAL_PATH, sep='\s+', header=None)
eval_df.columns = ['Images', 'Partition']
eval_df = eval_df.set_index('Images')

# merge attr_df and eval_df to get image data & its label in the same place
# and we can see which (image, label) belongs to which partition
merged_df = attr_df.merge(eval_df, left_index=True, right_index=True)

# write to csv depending on the partition
# the csv file will be read by custom dataset class
TRAIN_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-train.csv'
VAL_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-val.csv'
TEST_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-test.csv'

merged_df.loc[lambda df: df['Partition'] == 0].to_csv(TRAIN_CSV, columns=['Label'])
merged_df.loc[lambda df: df['Partition'] == 1].to_csv(VAL_CSV, columns=['Label'])
merged_df.loc[lambda df: df['Partition'] == 2].to_csv(TEST_CSV, columns=['Label'])