import get_data
import data_prep


def main():
    get_data.get_data()  # Load the data
    data_prep.prep_qa()  # Normalize the Q&A and create json & vocabularies


if __name__ == "__main__":
    main()
