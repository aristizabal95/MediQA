import re
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from mediqa.config.core import DATASET_DIR, config


def store_dataset(dset_name, dset_loc):
    dset = load_dataset(dset_name)
    for i in tqdm(range(len(dset["train"]))):
        title = dset["train"]["page_title"][i]
        title = re.sub(r'[^\w\s]', '-', title)
        contents = dset["train"]["page_text"][i]
        entry_path = Path(dset_loc / f"{title}.txt")
        if entry_path.exists():
            continue
        with open(entry_path, "w") as f:
            f.write(contents)


if __name__ == "__main__":
    dset_name = config.data_config.knowledge_dset
    dset_loc = config.data_config.knowledge_path

    dset_loc = Path(DATASET_DIR / dset_loc)
    dset_loc.mkdir(parents = True, exist_ok=True)

    store_dataset(dset_name, dset_loc)