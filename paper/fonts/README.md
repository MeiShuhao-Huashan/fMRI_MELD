# Fonts (Publication)

Figures are generated with Matplotlib. For submission-quality typography, we target **Arial**.

This repo does **not** ship Arial due to licensing. Use one of these options:

1) **Recommended (no admin needed): provide a local Arial TTF**
   - Copy a licensed `Arial.ttf` (e.g., from Windows: `C:\Windows\Fonts\arial.ttf`)
   - Place it at `paper/fonts/Arial.ttf` (auto-detected), or export:
     - `export MELD_PAPER_FONT_TTF=/path/to/Arial.ttf`

2) **Conda/Mamba (no sudo; proprietary license)**
   - Install Microsoft core fonts via conda-forge:
     - `mamba install -c conda-forge -y mscorefonts`
   - The figures auto-detect `$CONDA_PREFIX/fonts/arial.ttf` after installation.

3) **System install (Ubuntu; requires sudo)**
   - `sudo apt-get update && sudo apt-get install -y ttf-mscorefonts-installer`
   - `fc-cache -f -v`

If Arial is unavailable, the scripts fall back to `DejaVu Sans` and print a warning.
