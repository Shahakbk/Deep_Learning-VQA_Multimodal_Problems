"""
    Contains all the static definitions required for the code to run. Every changeable parameter will be set here.
"""

# Paths definitions
zip_dir = 'data/'  # Base data directory where the zips are downloaded to
qa_dir = 'data/QA/'  # Directory containing the questions and annotations jsons
images_dir = 'data/Images/'  # Directory containing the images jsons
vocabulary_dir = 'data/vocabulary.json'  # Directory where the created vocabularies jsons for Q&A are saved to
preprocessed_dir = 'data/resnet-14x14.h5'  # Directory where preprocessed features are saved to
train_path = 'data/Images/Train/train2014/'  # Path to train images directory
val_path = 'data/Images/Validation/val2014/'  # Path to validation images directory
#test_path = 'data/Images/Test/test2015/'  # Path to test images directory
#pretrained_path = 'data/resnet152-caffe.pth'  # Path to pre-trained ResNet-152 model
#pretrained_path = 'data/resnet34.pth'  # Path to pre-trained ResNet-34 model
pretrained_path = 'data/resnet18.pth'  # Path to pre-trained ResNet-18 model

paths = {"zip_dir": zip_dir,
         "qa_dir": qa_dir,
         "images_dir": images_dir,
         "vocabulary_dir": vocabulary_dir,
         "preprocessed_dir": preprocessed_dir,
         "train_path": train_path,
         "val_path": val_path,
         #"test_path": test_path,
         "pretrained_path": pretrained_path
         }

# Training parameters
preprocess_batch_size = 64
batch_size = 256
data_workers = 8
image_size = 448  # Scale the shorter end of an image to this size and centre crop
output_size = image_size // 32  # The size of the feature maps after processing through a network
output_features = 512  # The number of feature maps thereof
central_fraction = 0.875  # The amount to take from the image centre when cropping with centre crop
max_answers = 3000  # Hyper parameter for the answers vocabulary creation
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
epochs = 50

parameters = {"preprocess_batch_size": preprocess_batch_size,
              "batch_size": batch_size,
              "data_workers": data_workers,
              "image_size": image_size,
              "output_size": output_size,
              "output_features": output_features,
              "central_fraction": central_fraction,
              "max_answers": max_answers,
              "initial_lr": initial_lr,
              "lr_halflife": lr_halflife,
              "epochs": epochs
              }
