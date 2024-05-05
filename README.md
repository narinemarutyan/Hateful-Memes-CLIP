# Solving The Hateful Memes Challenge through Cross-Modal Interaction of CLIP Features

<img src='readme_images/hateful_memes.png' width=1501>

### Data Format

The Structure Of The Data Should Look Like This:
```
data/
├── hateful_memes
|                ├──img/
│                     ├── 1254.png
│                     ├── 1377.png
│                     └── ...
                ├── hateful_memes_expanded.csv



poetry install --group dev
```
## Preprocessing

### OFA-sys
```bash
git clone git@github.com:OFA-Sys/OFA.git
```
```bash
python preprocess:main
```
