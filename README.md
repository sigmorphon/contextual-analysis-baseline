# contextual-analysis-baseline

Our non-neural baseline for Task 2 of the shared task: morphology in context

## Setup

To initialize the repo to a usable state, run:

```bash
git clone git@github.com:sigmorphon/contextual-analysis-baseline.git
git submodule update --init --recursive
./lib/lemmingatize/setup.sh
```

## Annotate

For example, to decode the example model pretrained on `UD_Akkadian-PISANDUB` to a file named `akk-dev-predicted.conllu`:

```bash
./lemming annotate --marmot_model=models/UD_Akkadian-PISANDUB/model.marmot --lemming_model=models/UD_Akkadian-PISANDUB/model.lemming --input_file=data/UD_Akkadian-PISANDUB/akk_pisandub-um-dev.conllu --pred_file=akk-dev-predicted.conllu
```

