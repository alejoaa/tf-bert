#!/usr/bin/env python

"""
This script convers the a Jupyter Notebook to a Python script.
It removes cells with the tag 'remove_cell'.
    - input: Jupyter Notebook file
    - output: Python script
"""

# Imports
import argparse
import os
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter, PythonExporter, ScriptExporter, TemplateExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from traitlets.config import Config

# Create and setup argument parser
parser = argparse.ArgumentParser(
    prog="Convert Jupyter Notebook to Python Script",
    description="Placeholder"
)

parser.add_argument("-i", "--input-file", type=str, help="Jupyter Notebook to use as input", required=True)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-o", "--output-file", type=str, help="File store the output")
group.add_argument("-d", "--output-dir", type=str, help="Path store the output")
parser.add_argument("-t", "--template", type=str, help="nbconvert template to use")

args = parser.parse_args()

# Read arguments
input_file = args.input_file
template = args.template if args.template is not None else "python"

if args.output_file is not None:
    output_file = args.output_file
else:
    output_file = os.path.join(args.output_dir, Path(input_file).stem + ".py")

# Read content of input file (Jupyter Notebook)
with open(input_file) as f:
    lines = f.read()

notebook = nbformat.reads(lines, as_version=4)


# Configure and run out exporter
c = Config()
c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",) # The comma at the end is neccesary for this command to work
c.TemplateExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]


# TemplateExporter.extra_template_basedirs=["./"]
if args.template is not None:
    TemplateExporter.extra_template_paths=[args.template]

# Create exporter instance
exporter = PythonExporter(config = c, template_name=template)
exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)

# Run exporter
print(f"Converting notebook {input_file} to script")
print(f"** Using template: {template} **")
(body, resources) = exporter.from_notebook_node(notebook)

# Write output
with open(output_file, "w") as out:
    out.write(body)

print(f"Done. Script is located at {output_file}")