import pycad
from pycad.converters import NiftiToNrrdConverter

# Initialize converter
converter = NiftiToNrrdConverter()

# Specify the path to the input NIFTI file or directory containing multiple files
input_path = "path/to/input/file/or/dir"

# Specify the output directory to store the converted NRRD files
output_dir = "path/to/output/directory"

# Perform the conversion
converter.convert(input_path, output_dir)