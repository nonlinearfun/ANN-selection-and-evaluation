# ANN-selection-and-evaluation

The code accompanies the manuscript: Characterizing Evaporation Ducts Within the Marine Atmospheric Boundary Layer Using Artificial Neural Networks. Hilarie Sit and Christopher Earls.

This repository contains data, code and results for ANN model selection and model evaluation. Implementation using Tensorflow in Python.

## Perform model selection
Model selection is performed by grid search over the restricted hyperparameter space. 
To run the model selection code, specify name of csv data file and number of epochs for cases 1-3:
```bash
python model_selection_grid.py --csv case1 --n_epochs 1000
```
For cases 4-6, include the noise tag and specify number of augmented examples per training example (defaults shown)
```bash
python model_selection_grid.py --csv case1 --n_epochs 1000 --noise --n_aug 100
```

## Perform model evaluation
To run the model evaluation code, specify name of csv data file and hyperparameters corresponding to the selected model. Again, add noise and specify n_aug for cases 4-6.

```bash
python eval_neuralnetwork.py --csv case1 --n_epochs 1000 --n1 100 --bs 16 --lr 1e-4
```

## Results
All results from manuscript are included in the ANN_selection_results and ANN_evaluation_results folders.

To view loss and accuracy graphs, run tensorboard pointing to the correct log directory:
```bash
cd results
tensorboard --logdir=modelselectioncase1/logs
```