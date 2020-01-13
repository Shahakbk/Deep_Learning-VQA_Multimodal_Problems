"""
    Contains all the static definitions required for the code to run. Every changeable parameter will be set here.
"""

# Paths definitions
zip_dir = 'data/'  # Base data directory where the zips are downloaded to
qa_dir = 'data/QA/'  # Directory containing the questions and annotations jsons
images_dir = 'data/Images/'  # Directory containing the images jsons
vocabulary_dir = 'vocabulary.json'  # Directory where the created vocabularies jsons for Q&A are saved to


# Training parameters
max_answers = 3000  # Hyper parameter for the answers vocabulary creation
