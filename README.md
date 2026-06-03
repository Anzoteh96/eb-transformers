# Empirical Bayes, transformers, and Bayesian inference. 

This repo is based on the following two papers: 

- [Solving empirical Bayes via Transformers](https://arxiv.org/abs/2502.09844). 
- [Universal priors: solving empirical Bayes via Bayesian inference and pretraining](https://arxiv.org/abs/2602.15136). Accepted to COLT, 2026. 

## Selected Models

The selected models are stored with their hyperparameters in `./selected_models/`.
To read the pickle files on machines that do not have a GPU you use the function `load_model_dict` from `./eb_arena_mapper.py`, using the signature 
`load_model_dict([model_path], 'cpu')`. 

## Required libraries. 
These should be inside `requirements.txt` (other than the standard ones). 

## Training Details

The training process is implemented in `eb_train.py`. In particular: 

- For the neural-dirichlet mixture, we use the following settings. 

  ```
  python3 eb_train.py --nohist_thetas --train_steps 50000 --dmodel 32 --batch 200 --seqlen 512 --weight_share 0 --prior mixture --alpha 50 --dirich_prob 0.5 --layers 24 --heads 8 --train_lr 0.02 --train_lr_epoch 300 --train_lr_gamma 0.97 --decoding_layer_norm --theta_max 50.0 --theta_max_israndom 
  ```
  Note that `theta_max=50.0` is only a proxy of actual `theta_max` as we set `theta_max` to be random here. 

- For the universal prior (discrete/multinomial prior with random atom locations), we use the following settings. 
  
  ```
  python3 eb_train.py --nohist_thetas --train_steps 50000 --dmodel 32 --batch 200 --seqlen 512 --weight_share 0 --prior multinomial --atoms_locations uniform --num_atoms 10 --multin_dirich_param 1.0 --train_lr_epoch 300 --train_lr_gamma 0.99 --decoding_layer_norm --theta_max 50.0
  ```
  In particular, we choose 10 atoms following uniform distribution in [0, 50], and weights following Dirichlet(1, ..., 1). 

For attention-only setting, just add `--attn_only` flag. For parameter sharing, change the `weight_share` flag to `N` where `N` is the number of distinct weights. 


## Evaluation Details

### Synthetic Data

To evaluate on synthetic data we need two steps:

1. run `eb_area_mapper.py`. This produces Losses for a specific model on data generate by the provided model, you can also provide the name of other eb_estimators listed in it such as Robbins, ERM or NPNLE. 

This file uses the provide seed to create a list of random seeds. Then uses the argument `start` and `end` to take a slice of the seeds and run on them. (start =5 and end = 10 would run on 5 different seeds).  
  
```
python eb_arena_mapper.py --model [path to model or name of other estimator] --seed [integer to use for data generation] --prior [prior_type] --start [index of first seed] --end [index of last seed] --llmap_out [desired output dir]
```
We use ``--seed 684 --start 0 --end 4096`` evaluated on GPU to produce the plots in the paper. For example, on our T24r model we would do
```
python3 eb_arena_mapper.py --model selected_models/T24r.pkl --seed 684 --prior multinomial --start 0 --end 4096 --prior neural --theta_max 50 --seqlen 512 --batch 4000
```
to evaluate on multinomial prior-on-priors, and `--prior neural` to evaluate on neural priors. To evaluate on worst-case prior (only available for [0, 50]), pass in `--prior  worst_prior` and the `--worst_prior` flag. 


2. run `eb_arena_reducer.py`. This aggregates the losses from all the given model, and produces TSTAT / Plackett Luce coeffecients. It takes an input directory containing all the output files from previous steps, and adds `results.csv` and `tstats.csv`

```
python eb_arena_reducer.py --input_dir [dir with step_1_outputs]
```

### Comparing with Bayes estimator. 

Computation of Bayes estimator is supported on both the neural and multinomial prior-on-priors. 
This can be done by passing in `--model bayes` when running `eb_arena_mapper.py`. 

Computation of worst-case prior is only for [0, 50], in which case we hard code the minimum MSE as 25.179299501778658 (estimated via gradient descent when locating the worst prior). 


## Probing Details

In this section we investigate how do we do linear probe. Precisely, we need to first generate inputs, get the related attributes, and then the output of a trained model after each layer. 

### Input format. 
The inputs should be stored as pickle file of dictionary. It must contain the field `x` which is the inputs of the Poisson model, with dimension `B x N x 1` (N is the sequence length). 

### Getting attributes. 

The attributes `freqn, freqnplusone, erm` need to precomputed, via the following: 

`python3 get_attribute.py --input [input.pkl] --attribute [attribute] --output [where you want to store them]`
where `input.pkl` denotes the input file you stored from previous step, `[attribute]` is one of `erm, freqn, freqnplusone`.  

### Getting attributes (posterior density). 

Attribute involving the posterior density (as estimated by NPMLE) is more computationally intensive and needs to be done separately (prefarably in baches, and in parallel). To do so, you may use 
`python3 posterior_compute.py --filename [input.pkl] --start [a] --end [b] --outdir [posterior_dir]` 
where `[a]` and `[b]` are the start (inclusive) and end (exclusive) of the input slice. 
(E.g. a=0 and b=10 will give you the posterior density of `x[0:10]`). 

### Storing attributes. 
These attributes (from the same input pkl file) should be inside a same directory, with the name as subdirectory. E.g. it should be of the form 
```
[your_attr_dir]/[attr_name]/[blah].pkl
```
where attr_name can be erm, freqn, freqnplus1, posterior, posterior_next. 


### Linear Probe with GeLU

This trains and evaluates a decider to predict attributes with the activations. You might similary need to change attributes list. 
```
python3 train_layer_decoder.py --inputactivations [input.pkl] --outputname [whereyouwanttostore]  --attributedir [your_attr_dir] --model_path [saved_path] --lyr_num [0-24]
```
E.g. to get the linear probe result from the 10-th layerof T24r: 
```
python3 train_layer_decoder.py --inputactivations input.pkl --outputname linear_probe/lyr10.pkl  --attributedir my_attributes/ --model_path selected_models/T24r.pkl --lyr_num 10
```


## Curating and Preparing Datasets

### Baseball

Raw data obtained [here](https://www.retrosheet.org/game.htm) processed using  [this API](https://github.com/calestini/retrosheet). Save the files into `datasets/baseball`. 
Then, run `baseball_preprocess.py` (renaming variables `a` and `b` to your choice), and then run the following: 

```
python baseball_data.py --model [model] --pos [bat or pitch] --data_dir' [path to data] --year [year] --out_dir [results_dir]'
```
where `[results_dir]` is the directory you want to store all your results. 
The results will be stored as `[model]_[pos]_[year].pkl`. E.g. `T24r.pkl_bat_2019.pkl` for batting prediction done by T24r. 
This has fields inputs, outputs, and labels. 

To display the results after running all the models, simply call 
```
python baseball_reducer.py --results_dir [results_dir]
```


### Hockey
Raw data are obtain [here](https://www.hockey-reference.com/leagues/NHL_2019_skaters.html) (for the 2018-2019 season; the rest may be proceeded similarly). 
The files should be stored as `datasets/hockey/season_2019.csv`. After that, we run the program: 
```
python hockey_data.py --model [model] --data_dir datasets/hockey --prev_year [X] --next_year [X + 1]
```
to predict year X+1's goal based on year X. 
The results will be stored as `[model]_[pos]_[year1]_[year2].pkl`. E.g. `T24r.pkl_2018_2019.pkl` for prediction of all positions from 2018 input and 2019 label for T24r. `pos` field is empty for `all`, while present for defender, center, or winger. 
This has fields inputs, outputs, and labels. 

To display the results after running all the models, simply call 
```
python hockey_reducer.py --results_dir [results_dir] --pos [pos]
```
where `[pos]` denotes either `all, defender, center, winger`. 
This will print the p-value of the transformer vs the classical ones. 

### BookCorpusOpen. 

Step 1: download the dataset at `https://huggingface.co/datasets/lucadiliello/bookcorpusopen` and download into `bookcorpusopen` directory. 

Step 2: convert parquet file into 17868 files of books by `python bookcorpus_step0.py`, should save to `bookcorpusopen/files` directory.  

Step 3: Now get the frequency of each token in (X, Y) pair via `python bookcorpus_preprocess.py`. 

Step 4: Estimate via `python bookcorpus_estimate.py --model [model] --filename [filename] --tokenizer countvec`. For example, we may do 
```
python3 bookcorpus_estimate.py --model selected_models/T24r.pkl --dataset_dir bookcorpus/dataframes_countvec --filename 1-2-this-is-only-the-beginning --tokenizer countvec --out_dir hello
```
Note that filename is the root of the name of the book. 

## Hierarchical Bayes comparison. 

In this exercise, we numerically justify that a transformer pretrained on a simple PoP closely approximates the hierarchical Bayesian estimator, and fractional posterior in case train and test sequence lengths are different. This can be done by pretraining a transformer on a prior-on-priors (PoP) containing only a small number of priors. 

### Format of priors. 

Let `L` be the number of priors. Then this should be a pickle file of a dictionary containing the following fields: 

- `atoms`: list of length L, each being a numpy array of size `M x 1`. 
- `probs`: list of length L, each being a numpy array of size `M`. 


### Training. 

This is in the file `eb_train_fixedpriors.py`, e.g. using the following settings. 

```
python3 eb_train_fixedpriors.py --train_steps 200000 --dmodel 32 --batch 200 --seqlen [N] --theta_max 20 --weight_share 0 --prior_file [prior_file] --layers 24 --heads 8 --train_lr 0.02 --train_lr_epoch 300 --train_lr_gamma 0.99 --decoding_layer_norm --save_dir [savedir]
```

where [N] is your desired training sequence length (we played around with training sequence length), [prior_file] is the pickle file of priors, and savedir is where you want to save your model. 

### Evaluation. 

This is via the `hierarchical.py` script, e.g. with the following commands. 

```
python3 hierarchical.py --model [pickle file of transformer model] --test_seqlen [list of test sequence lengths] --csv_file [where_to_store_csv] --batch [num_batches]
```
Again, `--shrink_input_bincount` can be added to accelerate the evaluation process. 

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

```
@article{cannella2026universal,
  title={Universal priors: solving empirical Bayes via Bayesian inference and pretraining},
  author={Cannella, Nick and Teh, Anzo and Han, Yanjun and Polyanskiy, Yury},
  journal={arXiv preprint arXiv:2602.15136},
  year={2026}
}
```


