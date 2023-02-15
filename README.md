# bayesian-scores-pc
This repository contains the code for the paper **Bayesian Structure Scores for Probabilistic Circuits** (*AISTATS 2023*).

**Authors**: Yang Yang*, Gennaro Gala*, Robert Peharz

**Summary**: In this paper, we develop Bayesian structure scores for deterministic PCs, i.e., the structure likelihood with parameters marginalized out, 
which are well known as rigorous objectives for structure learning in probabilistic graphical models.
When used within a greedy cutset algorithm, our scores effectively protect against overfitting and yield a fast and almost hyper-parameter-free structure learner, distinguishing it from previous approaches.
In experiments, we achieve good trade-offs between training time and model fit in terms of log-likelihood.
Moreover, the principled nature of Bayesian scores unlocks PCs for accommodating frameworks such as structural expectation-maximization.

## Requirements

The required packages are listed in `requirements.txt`.

## Data

Please check https://github.com/UCLA-StarAI/Density-Estimation-Datasets for a collection of datasets used in the density estimation experiments.

## Usage

To run experiments for cutset learners, check the scripts in the `experiments` folder.

To learn a CNet using the BD score, you can call

```shell
python cnet_bd.py --ess 0.1 --n-search-ors 10 --learn-clt MI
```

```shell
usage: cnet_bd.py [-h] [--ess ESS] [--n-search-ors N_SEARCH_ORS] [--learn-clt {MI,BD}] [--output OUTPUT]

Original CNet Experiments

options:
  -h, --help            show this help message and exit
  --ess ESS             Equivalent sample size
  --n-search-ors N_SEARCH_ORS
                        Number of OR candidates
  --learn-clt {MI,BD}   Method of learning CLTs
  --output OUTPUT       Output store path
```

To learn a CNet using the BIC score, you can call

```shell
python cnet_bic.py --alpha 0.01 --n-search-ors 10 --learn-clt MI
```

```shell
usage: cnet_bic.py [-h] [--alpha ALPHA] [--n-search-ors N_SEARCH_ORS] [--learn-clt {MI,BD}] [--output OUTPUT]

CNetBIC Experiments

options:
  -h, --help            show this help message and exit
  --alpha ALPHA         Laplace smoothing factor
  --n-search-ors N_SEARCH_ORS
                        Number of OR candidates
  --learn-clt {MI,BD}   Method of learning CLTs
  --output OUTPUT       Output store path
```

To learn a mixture model, you can call

```shell
python3 em_bcnet.py nltcs -k 10 --ess 0.1 --n-search-ors 10 --learn-clt MI --n-iterations 20 --patience 3
```

```shell
usage: em_bcnet.py [-h] [-k K [K ...]] [--ess ESS] [--n-search-ors N_SEARCH_ORS] [--learn-clt {MI,BD}] [--n-iterations N_ITERATIONS] [--patience PATIENCE]
                   [--store-all-models] [--output OUTPUT]
                   {nltcs,msnbc,kdd,plants,baudio,jester,bnetflix,accidents,tretail,pumsb_star,dna,kosarek,msweb,book,tmovie,cwebkb,cr52,c20ng,bbc,ad,binarized_mnist}

EM CNetBD Experiments

positional arguments:
  {nltcs,msnbc,kdd,plants,baudio,jester,bnetflix,accidents,tretail,pumsb_star,dna,kosarek,msweb,book,tmovie,cwebkb,cr52,c20ng,bbc,ad,binarized_mnist}
                        Dataset

options:
  -h, --help            show this help message and exit
  -k K [K ...]          Number of EM components
  --ess ESS             Equivalent sample size
  --n-search-ors N_SEARCH_ORS
                        Number of OR candidates
  --learn-clt {MI,BD}   Method of learning CLTs
  --n-iterations N_ITERATIONS
                        Number of iterations for one EM
  --patience PATIENCE   Maximum number of iterations with no significant improvement
  --store-all-models
  --output OUTPUT       Output store path
```

## Citation
If you find this work useful, please consider citing:
```
TBD
```
## Acknowledgements
We thank the Eindhoven Artificial Intelligence Systems Institute for its support.