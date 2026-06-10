import csv
import os
from pathlib import Path


#
#
#       Validates the structure and naming of all the files to ensure compatibility with the rest of the code
#
#       - dataset
#       | - [category]
#         | - audio
#         | | - metadata.csv
#         | | - [name].[mp3/wav]
#         | - ground truth
#           | - [name].txt
#
#


def validate_data_formatting(fix=True):

    path = Path('../dataset')
    if not path.exists():
        raise FileNotFoundError("Missing root directory named 'dataset' for all data")
    for directory in [d for d in os.listdir() if Path(d).is_dir()]:
        validate_category(Path(directory), fix)

    print('validation succesfull')


def validate_category(directory: Path, fix=True):

    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Category '{directory}' is not present inside the dataset directory")

    if directory.name[0] == '.':
        return

    audio = directory / Path('audio')
    ground_truth = directory / Path('ground truth')

    print(directory)

    if not audio.exists():
        raise FileNotFoundError(f"Category '{directory}' does not contain a dedicated audio directory")
    if not ground_truth.exists():
        raise FileNotFoundError(f"Category '{directory}' does not contain a dedicated ground truth directory")

    if len([f for f in os.listdir(audio) if f != 'metadata.csv']) != len(os.listdir(ground_truth)):
        raise FileNotFoundError(f"Category '{directory}' has an unequal amounts of audio and transcripts")

    gt_files = os.listdir(ground_truth)
    gt_names = [file.split('.')[0] for file in gt_files]
    gt_extensions = [file.split('.')[-1] for file in gt_files]

    for file in os.listdir(audio):
        if file == 'metadata.csv':
            continue

        name = file.split('.')[0]
        extension = file.split('.')[-1]

        if extension not in ['mp3', 'wav']:
            raise TypeError(f"file '{file}' should be of type .mp3 or .wav in category '{directory}'s audio directory")
        if '_' in file:
            raise NameError(f"file '{file}' contains a invalid character '_'")

        if name not in gt_names:
            raise FileNotFoundError(f"file '{file}' had no matching ground truth file")

    for i in range(len(gt_extensions)):
        if gt_extensions[i] != 'txt':
            raise TypeError(f"file {gt_files[i]} should be of type .txt in category '{directory}'s audio directory")

    generate_metadata(directory / Path('audio'), override=True)


def generate_metadata(directory: Path, override=False):

    if isinstance(directory, str):
        directory = Path(directory)

    if directory.exists() and not override:
        return

    metadata_path = directory / 'metadata.csv'
    if metadata_path.exists():
        if not override:
            return
        metadata_path.unlink()

    a_files = os.listdir(directory)
    a_names = [file.split('.')[0] for file in a_files]

    with open(metadata_path, 'w', newline='', encoding='utf-8') as f_w:
        header = ['file_name', 'transcript', 'name']
        writer = csv.DictWriter(f_w, fieldnames=header)
        writer.writeheader()
        for i in range(len(a_files)):
            writer.writerow({
                'file_name': a_files[i],
                'transcript': f"{a_names[i]}.txt",
                'name': a_names[i]
            })

validate_data_formatting()
# validate_category(Path('Test'))
