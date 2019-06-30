# GDP-Autorec
***Last edit : 30 June 2019***

Recommender system using [AutoRec](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf) (without knowledge graph), trained using custom [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset.
<br>

# Domain of problems
*Given a list of previously interacted items and list of un-interacted items, find the most probable item from the un-interacted ones*

# Contents
- data : contains dataset to use in training
    - **intersect-20m** : custom ml-20m dataset where only movies shows up in Ripple-Net's knowledge graph used.    
    - ml-1m : original movielens 1m dataset
- **log** : contains training result stored in single folder named after training timestamp.
- **test** : contains jupyter notebook used in testing the trained models
<!-- ---------------------------------------- -->
- `AutoRec.py` : source code for Autorec
- `data_preprocessor.py` : preprocess data before fed into for Autorec
- `logger.py` : log training result and save models
- `main.py` : script to run training 
<!-- ---------------------------------------- -->
- `Preprocessor.ipynb` : jupyter notebook to run preprocess (produce same result as data_preprocessor.py)

### Note
    *italic* means this folder is ommited from git, but necessary if you need to run experiments
    **bold** means this folder has it's own README, check it for detailed information :)

# How to run
1. Prepare the dataset and the preprocessed version (check section below this)
2. Run the training script
    ~~~
    python3 main.py
    ~~~

# Preparing 
## Installing dependencies 

    pip3 install -r requirements.txt

## Missing component that are required 
- `data/intersect-20m/ratings.csr` : csr matrix (row = user, col = items, value = ratings) created with intersect-20m ratings, this file can be created using this [jupyter notebook]().

## How to prepare data
There are several ways to preprocess data (both yield the same result):
1. Simply provide `data/intersect-20m/ratings.csr` and run `main.py`, the script will preprocess it before training begin.
2. Run `Preprocessor.ipynb`. (the preprocess step of `main.py` will be skipped).

Desired output : 
- `data/intersect-20m/preprocessed_autorec_dataset` (used in train)
- `data/intersect-20m/preprocessed_autorec_dataset_test` (used for testing only)

# Training
## How to change hyper parameter
There are several ways to do this :
1. Open `main.py` and change the args parser default value
2. run `main.py` with arguments required.

# Testing / Evaluation
## How to check training result
1. Find the training result folder inside `/log` (find the latest), copy the folder name.
2. Create copy of latest jupyter notebook inside `/test` folder.
3. Rename folder to match a folder in `/log` (for traceability purpose).
4. Replace `TESTING_CODE` at the top of the notebook.
5. Run the notebook

# Final result
| Metric             | Value       |
|--------------------|-------------|
| Average prec@10    | +- 0.08     |
| Diversity@10 n=10  | 0.11 - 0.20 |
| Evaluated on       | 13.5k users |

# Other findings
- Models tends to suggest generic items that are rated high by a large number of users.
- Autorec model is much faster to train & test (than LightFM & RippleNet), and also much more robust since the model is not user centred.

# Experiment notes
- Don't normalize the rating (1 to 5 => 0 to 1): Model to learn to ranks items and give same top suggestion for every user.

# Author
- Jessin Donnyson - jessinra@gmail.com

# Contributors
- Benedict Tobias Henokh Wahyudi - tobi8800@gmail.com
- Michael Julio - michael.julio@gdplabs.id
- Fallon Candra - fallon.candra@gdplabs.id