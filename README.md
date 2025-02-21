# Solving Empirical Bayes via Transformers

This work is from our [paper](https://arxiv.org/abs/2502.09844). 

## Selected Models

The selected models are stored with their hyperparameters in `./selected_models/`.
To read the pickle files on machines that do not have a GPU you use the function `load_model_dict` from `./eb_arena_mapper.py` 


## Training Details

The training process is implemented in `eb_train.py`. 


## Evaluation Details

### Synthetic Data

To evaluate on synthetic data we need two steps:

1. run `eb_area_mapper.py`. This produces Losses for a specific model on data generate by the provided model, you can also provide the name of other eb_estimators listed in it such as Robbins, ERM or NPNLE. 

This file uses the provide seed to create a list of random seeds. Then uses the argument `start` and `end` to take a slice of the seeds and run on them. (start =5 and end = 10 would run on 5 different seeds).  
  
```
python eb_arena_mapper.py --model [path to model or name of other estimator] --seed [integer to use for data generation] --start [index of first seed] --end [index of last seed] --llmap_out [desired output dir]
```
We use ``--seed 684 --start 0 --end 4096`` evaluated on GPU to produce the plots in the paper. 


2. run `eb_arena_reducer.py`. This aggregates the losses from all the given model, and produces TSTAT / Plackett Luce coeffecients. It takes an input directory containing all the output files from previous steps, and adds `results.csv` and `tstats.csv`

```
python eb_arena_reducer.py --input_dir [dir with step_1_outputs]
```


### Real Data

We do not provide evaluation code for real data, but we provide code to get them to x, y pairs that should make it straightforward to reproduce our results.

## Probing Details

### Getting activations

The file get activations can be used to store activations of a layer of the model in a file. You can run it as follows:


```
python --model [path to model] --seed [integer seed] --num_samples [number of eb instances to sample] --output_dir$ 
```

### Getting attributes

You can also run a script to include attributes of the seeds, like for npmle priors, these are also saved in file per seed. You might want to modify the rest of attributed depending on what you are interested in but you can run

```
pythin probe_attribute.py --inputdir [path to one of the activation files from previous step, pnly need one per moddel] --attribute [name of the attribute to compute] --output [directory to output into]
```


### Linear Probe

You might need to modify attributes list in the code depending on which you are looking for

```
python probe_activations.py --inputdir [directory including activations of different layers in the model]    --outputname [output directory]    --attributedir [directory with a sub directory for each attributes]
```

### Linear Probe with GeLU

This trains and evaluates a decider to predict attributes with the activations. You might similary need to change attributes list

```
python train_layer_decoder.py --inputactivations [path to a picke file that contains activations] --outputname [name of output dir] --ttributedir [same as linear probe] --feature" [activations or mlp_steps, or atten_steps] --model_path [path to model] --jobname []
```


## Curating and Preparing Datasets



### Baseball

Raw data obtained [here](https://www.retrosheet.org/game.htm) processed using  [this API](https://github.com/calestini/retrosheet). Save the files into `datasets/baseball`. 
Then, run `baseball_preprocess.py` (renaming variables `a` and `b` to your choice), and then run the following: 

```
python baseball_data.py --model [model] --pos [bat or pitch] --data_dir' [path to data] --year [year]'
```


### Hockey
Raw data are obtain [here](https://www.hockey-reference.com/leagues/NHL_2019_skaters.html) (for the 2018-2019 season; the rest may be proceeded similarly). 
The files should be stored as `datasets/hockey/season_2019.csv`. After that, we run the program: 
```
python hockey_data.py --model [model] --data_dir datasets/hockey --prev_year [X] --next_year [X + 1]
```
to predict year X+1's goal based on year X. 

### BookCorpusOpen. 

Step 1: download the dataset at `https://huggingface.co/datasets/lucadiliello/bookcorpusopen` and download into `bookcorpusopen` directory. 

Step 2: convert parquet file into 17868 files of books by `python bookcorpus_step0.py`, should save to `bookcorpusopen/files` directory.  

Step 3: Now get the frequency of each token in (X, Y) pair via `python bookcorpus_preprocess.py`. 

Step 4: Estimate via `python bookcorpus_estimate.py --model [model] --filename [filename] --tokenizer countvec`. 

## BibTeX Citation

If you use our work, we would appreciate using the following citations: 

```
@article{teh2025empirical,
      title={Solving Empirical Bayes via Transformers}, 
      author={Teh, Anzo and Jabbour, Mark and Polyanskiy, Yury},
      journal={arXiv preprint arXiv:2502.09844},
      year={2025}
}
```


