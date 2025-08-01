# dmdv_osk_chimera
This repository contains supplementary information for the manuscript "Protein sequence evolution underlies interspecies incompatibility of a cell fate determinant."

## Repository Contents
- `*.py` Python script for an assaying the radial localization of Oskar signal from fluorescence micrographs of _Drosophila_  embryos.
- `*.pdf` Sequences of _D. melanogaster–D. virilis oskar_ chimeras expressed in _D. melanogaster_

## Radial localization of Oskar signal
### Running the Code
1. Prepare a directory containing rotated image stacks (`.tif`) and corresponding Cellpose-generated masks (`.npy`).
2. Edit the script to specify the correct path to your sample directory.
3. Update `genotype_tags` to match your teh dataset's genotypes.
4. Run the script.

### Plots Generated
- Radial profiles across erosion iterations
- Boxplots comparing slope distributions between genotypes

### Requirements
This code uses:
- Python 3.8+
- `numpy`, `matplotlib`, `opencv-python`, `scikit-image`, `scipy`, `seaborn`

Install packages via:
pip install numpy matplotlib opencv-python scikit-image scipy seaborn

## Chimera sequences
The sequences of the _D. melanogaster–D. virilis oskar_ chimeras generated for this experiment are listed in the attached `.pdf` file.

### License
This repository is shared for academic use. Please cite the manuscript if used in publications. A formal license can be added on request.

## Contact
For questions or collaboration, please contact:
Emily L. Rivard (author) or Cassandra G. Extavour (corresponding author)

