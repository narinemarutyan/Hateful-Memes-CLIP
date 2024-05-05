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
poetry install --dev
```

## Preprocessing

### OFA-sys


## Fine Tuning CLIP
```bash
python -m src.main
```
