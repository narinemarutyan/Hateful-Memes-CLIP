# Solving The Hateful Memes Challenge through Cross-Modal Interaction of CLIP Features

<img src='readme_images/hateful_memes.png' width=1501>

### Data Format

The Structure Of The Data Should Look Like This:
```
└──data
    └──hateful_memes
        ├──img
            ├── 1254.png
            ├── 1377.png
            └── 
        └──hateful_memes_expanded.csv
```

## Setup 

To setup the poetry environment you should have poetry installed on your system and then run the following commands
```bash
poetry install
```
If you want to do the preprocessing you should install dev dependencies as well by running:
```bash
poetry install --with dev
```

## Preprocessing

### OFA-sys


## Fine Tuning CLIP
Example of command
```bash
python -m src.main --image_path data/hateful_memes/img --csv_path data/hateful_memes/hateful_memes_expanded.csv --image_size 224 --max_epochs -1 --batch_size 9 --lr 1e-4
```