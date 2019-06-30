# Intersect-20m
Custom ml-20m dataset derived from [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset, where only movies shows up in Ripple-Net's knowledge graph used.

## Missing component that are required 
- `ratings.csr` : csr matrix (row = user, col = items, value = ratings) created with intersect-20m ratings, this file can be created using this [jupyter notebook]().