# contextual-analysis-neural-baseline


Our neural baseline for Task 2 of the shared task: morphology in context

## Prerequisites

CoNLL-U Parser (https://github.com/EmilStenstrom/conllu) :  ```pip install conllu```

PyTorch, version 0.3.0.post4


## Training from Scratch

Helper scripts are provided in the `scripts/` folder to run experiments for all languages.

To run the baseline tagger for a treebank,

```
python -u baselineTagger.py --treebank_path TREEBANK_DIR --langs lang --batch_size 32 --model_type mono --model_path MODEL_DIRECTORY
```
Substitute `TREEBANK_DIR` with the base directory of the task2 data,  `lang` with the name of the treebank, for e.g., `Afrikaans-AfriBooms` and `MODEL_DIRECTORY` with the location of the directory where the model should be written.

We also provide code to perform jackknifing for eliminating exposure bias when training the lemmatizer.
To generate the jackknifed training data, provide the additional argument `--jackknifing` and use cat_training_data.sh to create the combined training file.

```
python -u baselineTagger.py --treebank_path TREEBANK_DIR --langs lang --batch_size 32 --model_type mono --model_path MODEL_DIRECTORY --jackknife
```
Alternatively, you may use scripts `scripts/gen_scripts_baseline.py` and `scripts/gen_scripts_baseline_jackknife.py` to run experiments for all languages.

## Decoding with Pretrained Model

You can run evaluation with the argument `--test`. Please provide the argument `dev_set` with the argument `--test` to run evaluation on the development set. For example,

```
python -u baselineTagger.py --treebank_path TREEBANK_DIR --langs lang --batch_size 32 --model_type mono --model_path MODEL_DIRECTORY --test
```

## Pretrained Models

Pretrained Models are available at https://www.dropbox.com/sh/8m3di3fbtvbci3c/AAD5Evdl7hs6HyIgFV0xXVW0a?dl=0 .
We also provide jackknifed training data and the baseline predictions on the development set.
