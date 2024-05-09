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
        └──data.csv
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

### Preprocessing format

#### Masked 
<img src='readme_images/org_masked.png' width=900> 

```bash
 python masked.py --json_dir /path/to/label/json 
                  --img_folder /path/to/images 
                  --saved_csv_path /path/to/output.csv 
                  --ofa_checkpoint /path/to/checkpoint.pt
```
if you want to make the image masked  

### OFA-sys
The structure to make the JSON labels to CSV with captions:
```
└──OFA
    └─caption.py 
```
```bash
 git clone https://github.com/OFA-Sys/OFA.git
 mv preprocess/caption.py OFA
python caption.py --json_dir  path/to/labels
                  --img_folder path/to/img
                  --saved_csv_path  path/to/saved/data.csv
                  -ofa_checkpoint  path/to/ofa_checkpoint.pt
```
if you want to caption the image run above code


### BLIP

```bash
python blip.py --json_dir /path/to/json --img_folder /path/to/image/folder
```
if you want to caption the image with blip

## Fine Tuning CLIP
Example of command
```bash
python -m src.main  --image_size 224 --max_epochs -1 --batch_size 120 --lr 1e-4
```
