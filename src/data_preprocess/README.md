# Data Preprocess
## Get the original EurLex datasets
- Download the original data from https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2937
- Run `python eurlex_html2text.py --mapping <path_to_mapping_csv> --html-dir <directory_with_html_files> --data-dir <directory_with_label_data> --output <output_pickle_path>` 

## Data preprocess for training different models
- Follow the instructions of each model, e.g. for XRTransformer https://github.com/amzn/pecos/tree/mainline/pecos/xmc/xtransformer
