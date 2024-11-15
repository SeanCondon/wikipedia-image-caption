import base64
import csv
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
from typing import Dict
from urllib.parse import urlparse
import torchvision.transforms as transforms


def parse_tsv(filename: str):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            print(row)  # Process each row as needed


def str_to_bool(s: str) -> bool:
    return s.lower() in ['true', '1', 't', 'y', 'yes']


@dataclass
class TrainData:
    language: str
    page_url: str
    image_url: str
    page_title: str
    section_title: str
    hierarchical_section_title: str
    caption_reference_description: str
    caption_attribution_description: str
    caption_alt_text_description: str
    mime_type: str
    original_height: int
    original_width: int
    is_main_image: bool
    attribution_passes_lang_id: bool
    page_changed_recently: bool
    context_page_description: str
    context_section_description: str
    caption_title_and_reference_description: str


TRAIN_DATA_FILE_TO_LOAD = 1
TRAIN_DATA_FILE_MAX = 5


def main():
    """
    This is the main function that loads the training data.
    """
    load_image_data()


def load_training_data():
    """
    this a module that loads the wikipedia caption dataset and preprocesses it.
    It opens the tab separated file and reads the image name and its caption.
    """

    training_data: Dict[int, TrainData] = {}

    """
    Load the training data from train-<x>-of-5.tsv file
    """

    for trainIdx in range(0, TRAIN_DATA_FILE_TO_LOAD):
        # filename = f'../data/train-{trainIdx}-of-5.tsv'
        filename = f'/tmp/train-{trainIdx:05d}-of-{TRAIN_DATA_FILE_MAX:05d}.tsv'

        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if row[0] == 'language':
                    continue
                if len(row) != 18:
                    print(f"Invalid row: {row}")
                    continue
                training_row = TrainData(
                    language=row[0],
                    page_url=row[1],
                    image_url=row[2],
                    page_title=row[3],
                    section_title=row[4],
                    hierarchical_section_title=row[5],
                    caption_reference_description=row[6],
                    caption_attribution_description=row[7],
                    caption_alt_text_description=row[8],
                    mime_type=row[9],
                    original_height=int(row[10]),
                    original_width=int(row[11]),
                    is_main_image=str_to_bool(row[12]),
                    attribution_passes_lang_id=str_to_bool(row[13]),
                    page_changed_recently=str_to_bool(row[14]),
                    context_page_description=row[15],
                    context_section_description=row[16],
                    caption_title_and_reference_description=row[17],
                )
                training_data[len(training_data)] = training_row
            print(f"Loaded {len(training_data)} rows from {filename}")


def load_image_data():
    """
    this a module that loads the wikipedia caption dataset and preprocesses it.
    It opens the file and reads the image url and base64 encoded image.
    """
    for imageFileIdx in range(0, TRAIN_DATA_FILE_TO_LOAD):

        filename = f'/tmp/test_image_pixels_part-{imageFileIdx:05d}.csv'

        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                parsed_url = urlparse(row[0])
                path_parts = parsed_url.path.split('/')

                # Decode the base64 string
                image_data = base64.b64decode(row[1])

                # Convert the binary data to an image
                image = Image.open(BytesIO(image_data))

                image.show()

                # Define the transform to convert the PIL image to a PyTorch tensor
                transform = transforms.ToTensor()

                # Apply the transform to the image
                tensor_image = transform(image)

                print(f"loaded image {path_parts[-1]} in to tensor {tensor_image.shape}")


"""TODO finish the rest"""


if __name__ == "__main__":
    main()
