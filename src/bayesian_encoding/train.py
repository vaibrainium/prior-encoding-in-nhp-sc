class CVSet(dj.Computed):
    """
    Separates the CV set into the training and the test sets.
    """
    definition = """
    -> CleanContrastSessionDataSet
    -> CVSeed
    -> CVConfig
    ---.
    train_index: longblob    # training indices
    test_index:  longblob    # testing indices
    """

    def _make_tuples(self, key):
        print('Working on ', key)
        seed = key['cv_seed']
        np.random.seed(seed)
        fraction = float((CVConfig() & key).fetch1('cv_fraction'))
        dataset = (CleanContrastSessionDataSet() & key).fetch_dataset()
        N = len(dataset)
        pos = np.arange(N)
        split = round(N * fraction)
        np.random.shuffle(pos)
        key['train_index'] = pos[:split]
        key['test_index'] = pos[split:]
        self.insert1(key)

    def fetch_datasets(self):
        assert len(self) == 1, 'Only can fetch one dataset at a time'
        dataset = (CleanContrastSessionDataSet() & self).fetch_dataset()
        train_index, test_index = self.fetch1('train_index', 'test_index')
        train_set = dataset[train_index]
        test_set = dataset[test_index]
        return train_set, test_set



def binnify(x, center=270, delta=1, nbins=61, clip=True):
    """
    Bin the dat into bins, with center bin at `center`. Each bin has width `delta`
    and you will have equal number of bins to the left and to the right of the center bin.
    The left most bin starts at bin number and the last bin at `nbins`-1. If `clip`=True,
    then data falling out of the bins would be assigned bin number `-1` to indicate that it
    is out of the range. Otherwise, the data would be assigned to the nearest edge bin. A data point
    x would fall into bin i if  bin_i_left <= x < bin_i_right

    Args:
        x: data to bin
        center: center of the bins
        delta: width of each bin
        nbins: number of bins
        clip: whether to clip data falling out of bin range. Defaults to True

    Returns:
        (xv, p) - xv is an array of bin centers and thus has length nbins. p is the bin assignment of
            each data point in x and thus len(p) == len(x).
    """
    p = np.round((x - center) / delta) + (nbins // 2)
    if clip:
        out = (p < 0) | (p >= nbins)
        p[out] = -1
    else:
        p[p < 0] = 0
        p[p >= nbins] = nbins -1

    xv = (np.arange(nbins) - nbins//2) * delta + center
    return xv, p
