@echo off
REM LaTeX build script for IEEE paper

echo Building IEEE paper...

REM Run pdflatex twice to resolve references
pdflatex -interaction=nonstopmode p300_bci_ieee_paper.tex
pdflatex -interaction=nonstopmode p300_bci_ieee_paper.tex

echo Done! Check p300_bci_ieee_paper.pdf

pause
