# IEEE Paper Compilation Guide

This directory contains the LaTeX source files for the IEEE paper on the P300 BCI project.

## Prerequisites

To compile this paper, you need to have a LaTeX distribution installed on your system:

### For Windows:
- Install MiKTeX (https://miktex.org/download) or TeX Live (https://tug.org/texlive/)

### For macOS:
- Install MacTeX (https://tug.org/mactex/)

### For Linux:
- Install TeX Live: `sudo apt-get install texlive-full` (Ubuntu/Debian)

## Compiling the Paper

### Option 1: Using the Build Script
1. Make sure you have run `generate_figures.py` to create the performance plots
2. Run the `build_paper.bat` script (Windows) or use the following commands:

```bash
pdflatex -interaction=nonstopmode p300_bci_ieee_paper.tex
pdflatex -interaction=nonstopmode p300_bci_ieee_paper.tex
```

### Option 2: Using a LaTeX Editor
You can also use a LaTeX editor like TeXstudio, Overleaf, or Visual Studio Code with LaTeX Workshop extension:
1. Open `p300_bci_ieee_paper.tex` in your editor
2. Build or compile the document using the editor's tools

## Troubleshooting

- If you get errors about missing packages, let your LaTeX distribution install them automatically
- If there are issues with figures, make sure all the images exist in the specified paths
- If the IEEEtran class is missing, you can download it from IEEE's website: https://www.ieee.org/conferences/publishing/templates.html

## File Structure

- `p300_bci_ieee_paper.tex`: The main LaTeX source file
- `generate_figures.py`: Python script to generate performance plots
- `figures/`: Directory containing generated plots
- `build_paper.bat`: Windows batch script to compile the paper

## Note on IEEE Class Files

The IEEEtran class files are not included in this repository as they are standard in most LaTeX distributions. If you're having trouble, you can download the official IEEE templates from their website.
