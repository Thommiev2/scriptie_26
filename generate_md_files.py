from pathlib import Path
import csv

path_output = Path('md_files')
path_output.mkdir(exist_ok=True)  # Ensure the output directory exists

# --- Processing ASR Output ---
path = Path('output/asr output')

for file_path in path.iterdir():
    if file_path.is_file():
        with open(file_path, encoding='utf-8') as f_r:
            reader = csv.DictReader(f_r)
            for row in reader:
                # Construct filename
                file_name = f"{row['model']}_{row['category']}_{Path(row['name']).stem}.md"

                # Open with 'w' mode to overwrite or create
                with open(path_output / file_name, 'w', encoding='utf-8') as f_w:
                    f_w.write(row['transcript'])

path_gt = Path('dataset')

for category in path_gt.iterdir():
    if not category.name.startswith('.') and not category.suffix == '.py':
        for file_path in (path_gt / category.name / 'ground truth').iterdir():
            md_filename = f"gt_{category.name}_{file_path.stem}.md"
            with open(path_output / md_filename, 'w', encoding='utf-8') as f_w:
                f_w.write(open(file_path).read())
                f_w.close()
