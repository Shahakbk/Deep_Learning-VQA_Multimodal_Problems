import os

zip_dir = 'data/'
questions_dir = 'data/Questions/'
annotations_dir = 'data/Annotations/'
images_dir = 'data/Images/'


def get_data():
    # Download the data
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P ' + zip_dir)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P ' + zip_dir)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P ' + zip_dir)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P ' + zip_dir)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P ' + zip_dir)
    os.system('wget http://images.cocodataset.org/zips/train2014.zip -P ' + zip_dir)
    os.system('wget http://images.cocodataset.org/zips/val2014.zip -P ' + zip_dir)
    os.system('wget http://images.cocodataset.org/zips/test2015.zip -P ' + zip_dir)

    # Unzip the data
    os.system('unzip ' + zip_dir + 'v2_Questions_Train_mscoco.zip -d ' + questions_dir)
    os.system('unzip ' + zip_dir + 'v2_Questions_Val_mscoco.zip -d ' + questions_dir)
    os.system('unzip ' + zip_dir + 'v2_Questions_Test_mscoco.zip -d ' + questions_dir)
    os.system('unzip ' + zip_dir + 'v2_Annotations_Train_mscoco.zip -d ' + annotations_dir)
    os.system('unzip ' + zip_dir + 'v2_Annotations_Val_mscoco.zip -d ' + annotations_dir)
    os.system('unzip ' + zip_dir + 'train2014.zip -d ' + images_dir)
    os.system('unzip ' + zip_dir + 'val2014.zip -d ' + images_dir)
    os.system('unzip ' + zip_dir + 'test2015.zip -d ' + images_dir)
