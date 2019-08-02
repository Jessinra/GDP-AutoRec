# GDP-Autorec
***Last edit : 30 June 2019***

Recommender system using [AutoRec](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf) (without knowledge graph), trained using custom [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset.
<br>

# Domain of problems
*Given a list of previously interacted items and list of un-interacted items, find the most probable item from the un-interacted ones*

# Contents
- `/data` : contains dataset to use in training
    - **`/intersect-20m`** : custom ml-20m dataset where only movies shows up in Ripple-Net's knowledge graph used.    
    - `/ml-1m` : original movielens 1m dataset
- **`/log`** : contains training result stored in single folder named after training timestamp.
- **`/test`** : contains jupyter notebook used in testing the trained models
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

# Preparing 
## Installing dependencies 

    pip3 install -r requirements.txt

## Where to download the dataset
You can download the intersect-20m dataset [here](https://github.com/Jessinra/GDP-KG-Dataset). 

*Note : Dataset is put on separate repository because it's shared among models.*

## Missing component that are required 
- `data/intersect-20m/ratings_re2.csv` : After download the dataset, unzip the `ratings_re2.zip` and put inside the same folder as other things downloaded.
- `data/intersect-20m/ratings.csr` : csr matrix (row = user, col = items, value = ratings) created with intersect-20m ratings, this file can be created using this [jupyter notebook](https://github.com/Jessinra/GDP-KG-Dataset/blob/master/Preprocess.ipynb) (Preprocess.ipynb inside the dataset).

## How to prepare data

Simply provide `data/intersect-20m/ratings.csr` and run `main.py`, the script will preprocess it before training begin.
s
Desired output : 
- `data/intersect-20m/preprocessed_dataset` (used in train)
- `data/intersect-20m/preprocessed_dataset_test` (used for testing only)

# How to run
Simply run the `main.py`
~~~
python3 main.py
~~~

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
| Evaluated on  |  Prec@10   |
|---------------|------------|
|    500 user   |   0.05300  |
|   1000 user   |   0.05130  |
|   5000 user   |   0.05162  |
|  13850 user   |   0.05147  |
|  25000 user   |   0.05124  |

| Evaluated on  | Distinct@10   | Unique items |
|---------------|---------------|--------------|
|     10 user   |    0.21000    |    21        |
|     30 user   |    0.06667    |    20        |
|    100 user   |    0.02000    |    20        |
|   1000 user   |    0.00310    |    31        |
|   3000 user   |    0.00077    |    23        |

# Other findings
- Models tends to suggest generic items that are rated high by a large number of users.
- Autorec model is much faster to train & test (than LightFM & RippleNet), and also much more robust since the model is not user centred.

# Pros
- Fast training time and predicting time
- Not user centered (model doesn't remember user)

# Cons
- Require input to be in the form of pivot table (row = user, col = items)
- Require big memory size -> solved by using batch processing
- The model need to be re-trained for every new item addition.
- Doesn't handle cold start, every new user will be given the same recommendation.

# Experiment notes
- Normalizing the rating from (1 to 5 => 0 to 1) will lead to weird behaviour of model where it learn to rank items and give same top suggestion for every user.

# Author
- Jessin Donnyson - jessinra@gmail.com

# Contributors
- Benedict Tobias Henokh Wahyudi - tobi8800@gmail.com
- Michael Julio - michael.julio@gdplabs.id
- Fallon Candra - fallon.candra@gdplabs.id 