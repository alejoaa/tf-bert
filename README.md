# TF-BERT: A fine-tuned BERT model for Named Entity Recognition of transcription factors and target genes in scientific literature

TF-BERT is a [BioMedBERT](https://doi.org/10.1145/3458754) model fine-tuned on a custom Named Entity Recognition dataset derived from [ExTRI](https://doi.org/10.1016/j.bbagrm.2021.194778). Specifically, ExTRI contains sentences with mentions of interactions between transcription factors (TFs) and their target genes (TGs). We used an existing fine-tuned BioMedBERT model and fuzzy matching to label words/tokens in ExTRI’s sentences for TF and TG entities, using the reported TFs and TGs from ExTRI for each sentence. Next, we fine-tuned a pre-trained BioMedBERT model on the labeled sentences to detect TF and TG classes.

## Project structure
- `configs/`: Configuration files with arguments for fine-tuning BERT with `notebooks/run_ner.ipynb` or `scripts/run_ner.py`.
- `data/`: Folder that contains datasets (not included in this project).
- `nbtemplate/run_ner_template`: Jinja template for converting `notebooks/run_ner.ipynb` into `scripts/run_ner.py` using `nbconvert`. This custom template appends code to the script version so it can accept a configuration file as a command-line argument.
- `notebooks/`: Jupyter notebooks for the processing of data and fine-tuning of BERT.
- `scripts/`: Python scripts for the processing of data and fine-tuning of BERT. Currently only contains `run_ner.py` which is generated from `notebooks/run_ner.ipynb` using `nbconvert`.
- `utils/`: Helper functions or scripts. `jupyter_to_python.py` is a custom command-line tool for converting a Jupyter Notebook to Python script using `nbconvert`.

## Processing training data
This project has been used to fine-tuned BioMedBERT on two datasets: a custom dataset derived from ExTRI sentences and the BC2GM dataset. Jupyter notebooks for processing these datasets are available in the `notebooks` folder. Process datasets will be made available in the future.

## Run fine-tuning
To fine-tune a model, you can use either `notebooks/run_ner.ipynb` into `scripts/run_ner.py` with a configuration file. The config should indicate the values of fields for the classes `ModelArguments`, `DataTrainingArguments
` (custom class), and `TrainingArguments` (from Hugging Face `transformers`). For the script version, you can run:

```
scripts/./run_ner.py configs/example.yaml
```

## Generating scripts from Jupyter Notebooks
To generate `scripts/run_ner.py` you can run:
```
utils/./jupyter_to_python.py -i notebooks/run_ner.ipynb -d scripts -t nbtemplate/run_ner_template
```

## Next steps
This is an ongoing project which will have a wider scope. Future advances and updates will be posted here.
