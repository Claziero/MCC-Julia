# MCC-Julia
Implementation of the MCC algorithm in Julia programming language.

## Installation
First install the latest version of Julia, available at [julialang.org]()
Installation is done by executing the following commands:
- `julia`
- `] activate .` (make sure you are in the cloned folder)
- `instantiate`

## Usage
To use it, you can start `driver.jl` with different commands:
- `enhance`, then specify an input folder with `-i` (containing labeled dataset of fingerprints with 8 samples per subject, with names `subject_sample.jpg`) and an output folder with `-o`
- `match`, then specify the enhanced images folder with `-i`, and the matrix output file as png with `-o`
- `verification`, then specify the input matrix file with `-i` and the output graphs folder with `-o`
- `identification`, then specify the input matrix file with `-i` and the output graph folder with `-o`

## Credits
This project is based on the paper by
    R. Cappelli, M. Ferrara and D. Maltoni, 
    "**Minutia Cylinder-Code: A New Representation and Matching Technique for Fingerprint Recognition**" 
    in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 32, no. 12, pp. 2128-2141, Dec. 2010.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details