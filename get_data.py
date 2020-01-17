import os
from config import paths

# Get the paths
zip_dir = paths["zip_dir"]
qa_dir = paths["qa_dir"]
images_dir = paths["images_dir"]


def get_data():
    # Download the data
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P ' + zip_dir)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P ' + zip_dir)
    #os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P ' + zip_dir)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P ' + zip_dir)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P ' + zip_dir)
    os.system('wget http://images.cocodataset.org/zips/train2014.zip -P ' + zip_dir)
    os.system('wget http://images.cocodataset.org/zips/val2014.zip -P ' + zip_dir)
    #os.system('wget http://images.cocodataset.org/zips/test2015.zip -P ' + zip_dir)

    # Unzip the data
    os.system('unzip ' + zip_dir + 'v2_Questions_Train_mscoco.zip -d ' + qa_dir)
    os.system('unzip ' + zip_dir + 'v2_Questions_Val_mscoco.zip -d ' + qa_dir)
    #os.system('unzip ' + zip_dir + 'v2_Questions_Test_mscoco.zip -d ' + qa_dir)
    os.system('unzip ' + zip_dir + 'v2_Annotations_Train_mscoco.zip -d ' + qa_dir)
    os.system('unzip ' + zip_dir + 'v2_Annotations_Val_mscoco.zip -d ' + qa_dir)
    os.system('unzip ' + zip_dir + 'train2014.zip -d ' + images_dir)
    os.system('unzip ' + zip_dir + 'val2014.zip -d ' + images_dir)
    #os.system('unzip ' + zip_dir + 'test2015.zip -d ' + images_dir)
