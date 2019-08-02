class ProcessedDataContainer:

    def __init__(self):

        self.rating = None
        self.mask_rating = None

        self.train_rating = None
        self.train_mask_rating = None

        self.test_rating = None
        self.test_mask_rating = None

        self.n_train_rating = None
        self.n_test_rating = None

        self.train_users_idx = None
        self.train_items_idx = None

        self.test_users_idx = None
        self.test_items_idx = None

    def slice(self, start_idx=None, end_idx=None):
        
        self.rating = self.rating[start_idx, end_idx]
        self.mask_rating = self.mask_rating[start_idx, end_idx]

        self.train_rating = self.train_rating[start_idx, end_idx]
        self.train_mask_rating = self.train_mask_rating[start_idx, end_idx]

        self.test_rating = self.test_rating[start_idx, end_idx]
        self.test_mask_rating = self.test_mask_rating[start_idx, end_idx]

        self._update_count()

    def _update_count(self):

        self.n_test_rating = len(self.test_rating.nonzero()[0])  # non zero
        self.n_train_rating = len(self.rating.nonzero()[0]) - self.n_test_rating  # non zero from rating - non zero eval rating