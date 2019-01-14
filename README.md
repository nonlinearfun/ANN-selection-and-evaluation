# ANN-selection-and-evaluation

The code accompanies the manuscript: Characterizing Evaporation Ducts Within the Marine Atmospheric Boundary Layer Using Artificial Neural Networks. Hilarie Sit and Christopher Earls.

This repository contains data, code and results for ANN model selection and model evaluation. Implementation using Tensorflow in Python.

## Perform model selection
Model selection is performed by grid search over the restricted hyperparameter space. 
To run the model selection code, specify name of csv data file and number of epochs (defaults shown) for cases 1-3:
```bash
python model_selection_grid.py --csv case1 --n_epochs 300
```
For cases 4-6, include the noise tag and specify number of augmented examples per training example (defaults shown)
```bash
python model_selection_grid.py --csv case1 --n_epochs 300 --noise --n_aug 200
```

## Perform model evaluation
To run the model evaluation code, specify name of csv data file and hyperparameters corresponding to the selected model (defaults shown). Again, add noise and specify n_aug for cases 4-6.

```bash
python eval_neuralnetwork.py --csv case1 --n_epochs 500 --n1 45 --batch_size 4 --lr 5e-4
```

## Results
All results from manuscript are included in results folder

To view loss and accuracy graphs, run tensorboard pointing to the correct log directory:
```bash
cd results
tensorboard --logdir=modelselectioncase1/logs
```