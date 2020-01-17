import get_data
import data_prep
import images_prep
import utils


def main():
    device = utils.check_cuda()  # Check for CUDA availability
    get_data.get_data()  # Load the data
    data_prep.prep_qa()  # Normalize the Q&A and create json & vocabularies
    images_prep.prep_images(device)  # Pre-process the images


if __name__ == "__main__":
    main()
