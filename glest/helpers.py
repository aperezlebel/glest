import numbers

import matplotlib as mpl
import numpy as np
from matplotlib import colors
from sklearn.neighbors import KNeighborsRegressor


def bins_from_strategy(n_bins, strategy, y_prob=None):
    """Define the bin edges based on the strategy.
    If `n_bins` is already an `array-like`, it is used as-is by
    converting it to a `ndarray`.
    Parameters
    ----------
    n_bins : int or array-like
        Define the discretization applied to `y_prob`, ranging in [0, 1].
        - if an integer is provided, the discretization depends on the
            `strategy` parameter with n_bins as the number of bins.
        - if an array-like is provided, the `strategy` parameter is overlooked
            and the array is used as bin edges directly.
    strategy : {'uniform', 'quantile'}
        Strategy used to define the widths of the bins.
        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.
        Ignored if `n_bins` is an array-like containing the bin edges.
    y_prob : array-like of shape (n_samples,), default=None
        Probabilities of the positive class. Used when `strategy='quantile'`.
    Returns
    -------
    bins : ndarray
        The bin edges. If `n_bins` is an integer, `bins` is of shape (n_bins+1,).
        If `n_bins` is an array-like, `bins` has same shape as `n_bins`.
    """
    if isinstance(n_bins, numbers.Real):
        if strategy == "quantile":
            # Determine bin edges by distribution of data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(y_prob, quantiles * 100)
            bins[0] = 0
            bins[-1] = 1
        elif strategy == "uniform":
            bins = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            raise ValueError(
                "Invalid entry to 'strategy' input. Strategy "
                "must be either 'quantile' or 'uniform'."
            )

    else:  # array-like
        bins = np.asarray(n_bins)

    return bins


def scores_to_bin_ids(y_scores, n_bins, strategy):
    """Get bin id from continuous scores.

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,)
        Probabilities of the positive class in [0, 1]. Probabilities
        outside of [0, 1] while be clipped to the nearest bins.
    n_bins : int or array-like
        Define the discretization applied to `y_scores`, ranging in [0, 1].
        - if an integer is provided, the discretization depends on the
            `strategy` parameter with n_bins as the number of bins.
        - if an array-like is provided, the `strategy` parameter is overlooked
            and the array is used as bin edges directly.
    strategy : {'uniform', 'quantile'}
        Strategy used to define the widths of the bins.
        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.
        Ignored if `n_bins` is an array-like containing the bin edges.

    Returns
    -------
    y_bins : ndarray of shape (n_samples,) and type int64
        Integers ranging between 0 and n_bins - 1
    bins : ndarray of shape (n_bins + 1,)
        The bin edges.
    """
    bins = bins_from_strategy(n_bins, strategy, y_scores)
    n_bins = len(bins) - 1  # bins are the bin edges

    # Get bin assignment for each sample
    y_bins = np.digitize(y_scores, bins=bins) - 1
    y_bins = np.clip(y_bins, a_min=None, a_max=n_bins - 1)

    return y_bins, bins


def list_list_to_array(L, fill_value=None, dtype=None):
    """Convert a list of list of varying size into a numpy array with
    smaller shape possible.

    Parameters
    ----------
    L : list of lists.

    fill_value : any
        Value to fill the blank with.

    Returns
    -------
    a : array

    """
    max_length = max(map(len, L))
    L = [Li + [fill_value] * (max_length - len(Li)) for Li in L]
    return np.array(L, dtype=dtype)


def _validate_clustering(*args):
    if len(args) == 2:
        frac_pos, counts = args
    elif len(args) == 3:
        frac_pos, counts, mean_scores = args
    else:
        raise ValueError(f"2 or 3 args must be given. Got {len(args)}.")

    if frac_pos.shape != counts.shape:
        raise ValueError(
            f"Shape mismatch between frac_pos {frac_pos.shape} "
            f"and counts {counts.shape}."
        )

    if len(args) == 3 and frac_pos.shape != mean_scores.shape:
        raise ValueError(
            f"Shape mismatch between frac_pos {frac_pos.shape} and "
            f"mean_scores {mean_scores.shape}."
        )

    if frac_pos.ndim < 2:
        raise ValueError(
            f"frac_pos, counts and mean_scores must bet at least "
            f"2D. Got {frac_pos.ndim}D."
        )


def check_2D_array(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    elif x.ndim == 2 and x.shape[1] != 1:
        raise ValueError(f"x must have one feature. Got shape " f"{x.shape}")

    elif x.ndim > 2:
        raise ValueError(f"x must be at most 2 dimensional. " f"Got shape {x.shape}")

    return x


class CEstimator:
    def __init__(self, y_scores, y_labels):
        y_scores = np.array(y_scores)
        y_labels = np.array(y_labels)
        y_scores = check_2D_array(y_scores)
        self.y_scores = y_scores
        self.y_labels = y_labels

    def _c_hat(self, test_scores):
        test_scores = check_2D_array(test_scores)
        n_neighbors = min(2000, int(0.1 * len(test_scores)))
        est = KNeighborsRegressor(n_neighbors=n_neighbors)
        est.fit(self.y_scores.reshape(-1, 1), self.y_labels)
        c_hat = est.predict(test_scores)
        return c_hat

    def c_hat(self):
        return self._c_hat(self.y_scores.reshape(-1, 1))


def psr_name_to_entropy(psr: str):
    """Get the entropy of a scoring rule.

    Parameters
    ----------
    psr : str | Callable
        The name of the scoring rule in {"brier", "log"}. Or its entropy
        given as a callable `lambda p: entropy(p)`.

    Returns
    -------
    Callable
        The entropy of the scoring rule.

    Raises
    ------
    ValueError
        If psr is neither a valid string nor a callable.

    """
    available_metrics = ["brier", "log"]

    if callable(psr):
        return psr

    elif psr == "brier":
        return lambda x: 2 * x * (1 - x)

    elif psr == "log":
        return lambda x: -(x * np.log(x) + (1 - x) * np.log(1 - x))

    else:
        raise ValueError(f'Unknown metric "{psr}". Choices: {available_metrics}.')


def compute_GL_induced(c_hat, y_bins, psr: str = "brier"):
    """Estimate GL induced for the Brier score."""

    uniques, counts = np.unique(y_bins, return_counts=True)
    diff = []

    entropy = psr_name_to_entropy(psr)

    for i in uniques:
        c_bin = c_hat[y_bins == i]
        d = entropy(np.mean(c_bin)) - np.mean(entropy(c_bin))
        diff.append(d)

    GL_ind = np.vdot(diff, counts) / np.sum(counts)

    return GL_ind


def filter_valid_counts(counts):
    """Discard regions with only one sample since the debiasing is not valid
    for this situation."""
    counts = counts.copy()
    counts[counts == 1] = 0
    return counts


def compute_GL_uncorrected(frac_pos, counts, psr: str = "brier"):
    counts = filter_valid_counts(counts)

    prob_bins = calibration_curve(
        frac_pos, counts, remove_empty=False, return_mean_bins=False
    )
    entropy = psr_name_to_entropy(psr)
    diff = entropy(prob_bins[:, None]) - entropy(frac_pos)

    n_samples = np.sum(counts)
    if n_samples > 0:
        return np.nansum(counts * diff) / n_samples
    else:
        return 0


def compute_GL_bias(frac_pos, counts, psr: str = "brier"):
    if psr != "brier":
        print('Warning: GL bias computation is only available for "brier" psr.')
        return np.nan

    counts = filter_valid_counts(counts)

    prob_bins = calibration_curve(
        frac_pos, counts, remove_empty=False, return_mean_bins=False
    )
    n_bins = np.sum(counts, axis=1)  # number of samples in bin
    n = np.sum(n_bins)
    var = np.divide(
        frac_pos * (1 - frac_pos),
        counts - 1,
        np.full_like(frac_pos, 0, dtype=float),
        where=counts > 1,
    )
    var = var * np.divide(
        counts,
        n_bins[:, None],
        np.full_like(frac_pos, 0, dtype=float),
        where=n_bins[:, None] > 0,
    )
    var2 = np.divide(
        prob_bins * (1 - prob_bins),
        n_bins - 1,
        np.full_like(prob_bins, 0, dtype=float),
        where=n_bins > 1,
    )
    bias = np.sum(var, axis=1) - var2
    bias *= np.divide(
        n_bins,
        n,
        np.full_like(n_bins, 0, dtype=float),
        where=n > 0,
    )
    bias *= 2  # for the Brier score
    bias = np.sum(bias)
    return bias


def calibration_curve(
    frac_pos, counts, mean_scores=None, remove_empty=True, return_mean_bins=True
):
    """Compute calibration curve from output of clustering.
    Result is the same as sklearn's calibration_curve.

    Parameters
    ----------
    frac_pos : (bins, n_clusters) array
        The fraction of positives in each cluster for each bin.

    counts : (bins, n_clusters) array
        The number of samples in each cluster for each bin.

    mean_scores : (bins, n_clusters) array
        The mean score of samples in each cluster for each bin.

    remove_empty : bool
        Whether to remove empty bins.

    return_mean_bins : bool
        Whether to return mean_bins.

    Returns
    -------
    prob_bins : (bins,) arrays
        Fraction of positives in each bin.

    mean_bins : (bins,) arrays
        Mean score in each bin. Returned only if return_mean_bins=True.

    """
    if return_mean_bins and mean_scores is None:
        raise ValueError("mean_scores cannot be None when " "return_mean_bins=True.")

    if not return_mean_bins:
        _validate_clustering(frac_pos, counts)

    else:
        _validate_clustering(frac_pos, counts, mean_scores)

    count_sums = np.sum(counts, axis=1, dtype=float)
    non_empty = count_sums > 0
    prob_bins = np.divide(
        np.sum(frac_pos * counts, axis=1),
        count_sums,
        where=non_empty,
        out=np.full_like(count_sums, np.nan),
    )

    if return_mean_bins:
        mean_bins = np.divide(
            np.sum(mean_scores * counts, axis=1),
            count_sums,
            where=non_empty,
            out=np.full_like(count_sums, np.nan),
        )

    # The calibration_curve of sklearn removes empty bins.
    # Should do the same to give same result.
    if frac_pos.ndim == 2 and remove_empty:
        prob_bins = prob_bins[non_empty]
        if return_mean_bins:
            mean_bins = mean_bins[non_empty]

    if return_mean_bins:
        return prob_bins, mean_bins

    return prob_bins


# Register flare colormap from Seaborn
_flare_lut = [
    [0.92907237, 0.68878959, 0.50411509],
    [0.92891402, 0.68494686, 0.50173994],
    [0.92864754, 0.68116207, 0.4993754],
    [0.92836112, 0.67738527, 0.49701572],
    [0.9280599, 0.67361354, 0.49466044],
    [0.92775569, 0.66983999, 0.49230866],
    [0.9274375, 0.66607098, 0.48996097],
    [0.927111, 0.66230315, 0.48761688],
    [0.92677996, 0.6585342, 0.485276],
    [0.92644317, 0.65476476, 0.48293832],
    [0.92609759, 0.65099658, 0.48060392],
    [0.925747, 0.64722729, 0.47827244],
    [0.92539502, 0.64345456, 0.47594352],
    [0.92503106, 0.6396848, 0.47361782],
    [0.92466877, 0.6359095, 0.47129427],
    [0.92429828, 0.63213463, 0.46897349],
    [0.92392172, 0.62835879, 0.46665526],
    [0.92354597, 0.62457749, 0.46433898],
    [0.9231622, 0.6207962, 0.46202524],
    [0.92277222, 0.61701365, 0.45971384],
    [0.92237978, 0.61322733, 0.45740444],
    [0.92198615, 0.60943622, 0.45509686],
    [0.92158735, 0.60564276, 0.45279137],
    [0.92118373, 0.60184659, 0.45048789],
    [0.92077582, 0.59804722, 0.44818634],
    [0.92036413, 0.59424414, 0.44588663],
    [0.91994924, 0.5904368, 0.44358868],
    [0.91952943, 0.58662619, 0.4412926],
    [0.91910675, 0.58281075, 0.43899817],
    [0.91868096, 0.57899046, 0.4367054],
    [0.91825103, 0.57516584, 0.43441436],
    [0.91781857, 0.57133556, 0.43212486],
    [0.9173814, 0.56750099, 0.4298371],
    [0.91694139, 0.56366058, 0.42755089],
    [0.91649756, 0.55981483, 0.42526631],
    [0.91604942, 0.55596387, 0.42298339],
    [0.9155979, 0.55210684, 0.42070204],
    [0.9151409, 0.54824485, 0.4184247],
    [0.91466138, 0.54438817, 0.41617858],
    [0.91416896, 0.54052962, 0.41396347],
    [0.91366559, 0.53666778, 0.41177769],
    [0.91315173, 0.53280208, 0.40962196],
    [0.91262605, 0.52893336, 0.40749715],
    [0.91208866, 0.52506133, 0.40540404],
    [0.91153952, 0.52118582, 0.40334346],
    [0.91097732, 0.51730767, 0.4013163],
    [0.910403, 0.51342591, 0.39932342],
    [0.90981494, 0.50954168, 0.39736571],
    [0.90921368, 0.5056543, 0.39544411],
    [0.90859797, 0.50176463, 0.39355952],
    [0.90796841, 0.49787195, 0.39171297],
    [0.90732341, 0.4939774, 0.38990532],
    [0.90666382, 0.49008006, 0.38813773],
    [0.90598815, 0.486181, 0.38641107],
    [0.90529624, 0.48228017, 0.38472641],
    [0.90458808, 0.47837738, 0.38308489],
    [0.90386248, 0.47447348, 0.38148746],
    [0.90311921, 0.4705685, 0.37993524],
    [0.90235809, 0.46666239, 0.37842943],
    [0.90157824, 0.46275577, 0.37697105],
    [0.90077904, 0.45884905, 0.37556121],
    [0.89995995, 0.45494253, 0.37420106],
    [0.89912041, 0.4510366, 0.37289175],
    [0.8982602, 0.44713126, 0.37163458],
    [0.89737819, 0.44322747, 0.37043052],
    [0.89647387, 0.43932557, 0.36928078],
    [0.89554477, 0.43542759, 0.36818855],
    [0.89458871, 0.4315354, 0.36715654],
    [0.89360794, 0.42764714, 0.36618273],
    [0.89260152, 0.42376366, 0.36526813],
    [0.8915687, 0.41988565, 0.36441384],
    [0.89050882, 0.41601371, 0.36362102],
    [0.8894159, 0.41215334, 0.36289639],
    [0.888292, 0.40830288, 0.36223756],
    [0.88713784, 0.40446193, 0.36164328],
    [0.88595253, 0.40063149, 0.36111438],
    [0.88473115, 0.39681635, 0.3606566],
    [0.88347246, 0.39301805, 0.36027074],
    [0.88217931, 0.38923439, 0.35995244],
    [0.880851, 0.38546632, 0.35970244],
    [0.87947728, 0.38172422, 0.35953127],
    [0.87806542, 0.37800172, 0.35942941],
    [0.87661509, 0.37429964, 0.35939659],
    [0.87511668, 0.37062819, 0.35944178],
    [0.87357554, 0.36698279, 0.35955811],
    [0.87199254, 0.3633634, 0.35974223],
    [0.87035691, 0.35978174, 0.36000516],
    [0.86867647, 0.35623087, 0.36033559],
    [0.86694949, 0.35271349, 0.36073358],
    [0.86516775, 0.34923921, 0.36120624],
    [0.86333996, 0.34580008, 0.36174113],
    [0.86145909, 0.3424046, 0.36234402],
    [0.85952586, 0.33905327, 0.36301129],
    [0.85754536, 0.33574168, 0.36373567],
    [0.855514, 0.33247568, 0.36451271],
    [0.85344392, 0.32924217, 0.36533344],
    [0.8513284, 0.32604977, 0.36620106],
    [0.84916723, 0.32289973, 0.36711424],
    [0.84696243, 0.31979068, 0.36806976],
    [0.84470627, 0.31673295, 0.36907066],
    [0.84240761, 0.31371695, 0.37010969],
    [0.84005337, 0.31075974, 0.37119284],
    [0.83765537, 0.30784814, 0.3723105],
    [0.83520234, 0.30499724, 0.37346726],
    [0.83270291, 0.30219766, 0.37465552],
    [0.83014895, 0.29946081, 0.37587769],
    [0.82754694, 0.29677989, 0.37712733],
    [0.82489111, 0.29416352, 0.37840532],
    [0.82218644, 0.29160665, 0.37970606],
    [0.81942908, 0.28911553, 0.38102921],
    [0.81662276, 0.28668665, 0.38236999],
    [0.81376555, 0.28432371, 0.383727],
    [0.81085964, 0.28202508, 0.38509649],
    [0.8079055, 0.27979128, 0.38647583],
    [0.80490309, 0.27762348, 0.3878626],
    [0.80185613, 0.2755178, 0.38925253],
    [0.79876118, 0.27347974, 0.39064559],
    [0.79562644, 0.27149928, 0.39203532],
    [0.79244362, 0.2695883, 0.39342447],
    [0.78922456, 0.26773176, 0.3948046],
    [0.78596161, 0.26594053, 0.39617873],
    [0.7826624, 0.26420493, 0.39754146],
    [0.77932717, 0.26252522, 0.39889102],
    [0.77595363, 0.2609049, 0.4002279],
    [0.77254999, 0.25933319, 0.40154704],
    [0.76911107, 0.25781758, 0.40284959],
    [0.76564158, 0.25635173, 0.40413341],
    [0.76214598, 0.25492998, 0.40539471],
    [0.75861834, 0.25356035, 0.40663694],
    [0.75506533, 0.25223402, 0.40785559],
    [0.75148963, 0.2509473, 0.40904966],
    [0.74788835, 0.24970413, 0.41022028],
    [0.74426345, 0.24850191, 0.41136599],
    [0.74061927, 0.24733457, 0.41248516],
    [0.73695678, 0.24620072, 0.41357737],
    [0.73327278, 0.24510469, 0.41464364],
    [0.72957096, 0.24404127, 0.4156828],
    [0.72585394, 0.24300672, 0.41669383],
    [0.7221226, 0.24199971, 0.41767651],
    [0.71837612, 0.24102046, 0.41863486],
    [0.71463236, 0.24004289, 0.41956983],
    [0.7108932, 0.23906316, 0.42048681],
    [0.70715842, 0.23808142, 0.42138647],
    [0.70342811, 0.2370976, 0.42226844],
    [0.69970218, 0.23611179, 0.42313282],
    [0.69598055, 0.2351247, 0.42397678],
    [0.69226314, 0.23413578, 0.42480327],
    [0.68854988, 0.23314511, 0.42561234],
    [0.68484064, 0.23215279, 0.42640419],
    [0.68113541, 0.23115942, 0.42717615],
    [0.67743412, 0.23016472, 0.42792989],
    [0.67373662, 0.22916861, 0.42866642],
    [0.67004287, 0.22817117, 0.42938576],
    [0.66635279, 0.22717328, 0.43008427],
    [0.66266621, 0.22617435, 0.43076552],
    [0.65898313, 0.22517434, 0.43142956],
    [0.65530349, 0.22417381, 0.43207427],
    [0.65162696, 0.22317307, 0.4327001],
    [0.64795375, 0.22217149, 0.43330852],
    [0.64428351, 0.22116972, 0.43389854],
    [0.64061624, 0.22016818, 0.43446845],
    [0.63695183, 0.21916625, 0.43502123],
    [0.63329016, 0.21816454, 0.43555493],
    [0.62963102, 0.2171635, 0.43606881],
    [0.62597451, 0.21616235, 0.43656529],
    [0.62232019, 0.21516239, 0.43704153],
    [0.61866821, 0.21416307, 0.43749868],
    [0.61501835, 0.21316435, 0.43793808],
    [0.61137029, 0.21216761, 0.4383556],
    [0.60772426, 0.2111715, 0.43875552],
    [0.60407977, 0.21017746, 0.43913439],
    [0.60043678, 0.20918503, 0.43949412],
    [0.59679524, 0.20819447, 0.43983393],
    [0.59315487, 0.20720639, 0.44015254],
    [0.58951566, 0.20622027, 0.44045213],
    [0.58587715, 0.20523751, 0.44072926],
    [0.5822395, 0.20425693, 0.44098758],
    [0.57860222, 0.20328034, 0.44122241],
    [0.57496549, 0.20230637, 0.44143805],
    [0.57132875, 0.20133689, 0.4416298],
    [0.56769215, 0.20037071, 0.44180142],
    [0.5640552, 0.19940936, 0.44194923],
    [0.56041794, 0.19845221, 0.44207535],
    [0.55678004, 0.1975, 0.44217824],
    [0.55314129, 0.19655316, 0.44225723],
    [0.54950166, 0.19561118, 0.44231412],
    [0.54585987, 0.19467771, 0.44234111],
    [0.54221157, 0.19375869, 0.44233698],
    [0.5385549, 0.19285696, 0.44229959],
    [0.5348913, 0.19197036, 0.44222958],
    [0.53122177, 0.1910974, 0.44212735],
    [0.52754464, 0.19024042, 0.44199159],
    [0.52386353, 0.18939409, 0.44182449],
    [0.52017476, 0.18856368, 0.44162345],
    [0.51648277, 0.18774266, 0.44139128],
    [0.51278481, 0.18693492, 0.44112605],
    [0.50908361, 0.18613639, 0.4408295],
    [0.50537784, 0.18534893, 0.44050064],
    [0.50166912, 0.18457008, 0.44014054],
    [0.49795686, 0.18380056, 0.43974881],
    [0.49424218, 0.18303865, 0.43932623],
    [0.49052472, 0.18228477, 0.43887255],
    [0.48680565, 0.1815371, 0.43838867],
    [0.48308419, 0.18079663, 0.43787408],
    [0.47936222, 0.18006056, 0.43733022],
    [0.47563799, 0.17933127, 0.43675585],
    [0.47191466, 0.17860416, 0.43615337],
    [0.46818879, 0.17788392, 0.43552047],
    [0.46446454, 0.17716458, 0.43486036],
    [0.46073893, 0.17645017, 0.43417097],
    [0.45701462, 0.17573691, 0.43345429],
    [0.45329097, 0.17502549, 0.43271025],
    [0.44956744, 0.17431649, 0.4319386],
    [0.44584668, 0.17360625, 0.43114133],
    [0.44212538, 0.17289906, 0.43031642],
    [0.43840678, 0.17219041, 0.42946642],
    [0.43469046, 0.17148074, 0.42859124],
    [0.4309749, 0.17077192, 0.42769008],
    [0.42726297, 0.17006003, 0.42676519],
    [0.42355299, 0.16934709, 0.42581586],
    [0.41984535, 0.16863258, 0.42484219],
    [0.41614149, 0.16791429, 0.42384614],
    [0.41244029, 0.16719372, 0.42282661],
    [0.40874177, 0.16647061, 0.42178429],
    [0.40504765, 0.16574261, 0.42072062],
    [0.401357, 0.16501079, 0.41963528],
    [0.397669, 0.16427607, 0.418528],
    [0.39398585, 0.16353554, 0.41740053],
    [0.39030735, 0.16278924, 0.41625344],
    [0.3866314, 0.16203977, 0.41508517],
    [0.38295904, 0.16128519, 0.41389849],
    [0.37928736, 0.16052483, 0.41270599],
    [0.37562649, 0.15974704, 0.41151182],
    [0.37197803, 0.15895049, 0.41031532],
    [0.36833779, 0.15813871, 0.40911916],
    [0.36470944, 0.15730861, 0.40792149],
    [0.36109117, 0.15646169, 0.40672362],
    [0.35748213, 0.15559861, 0.40552633],
    [0.353885, 0.15471714, 0.40432831],
    [0.35029682, 0.15381967, 0.4031316],
    [0.34671861, 0.1529053, 0.40193587],
    [0.34315191, 0.15197275, 0.40074049],
    [0.33959331, 0.15102466, 0.3995478],
    [0.33604378, 0.15006017, 0.39835754],
    [0.33250529, 0.14907766, 0.39716879],
    [0.32897621, 0.14807831, 0.39598285],
    [0.3254559, 0.14706248, 0.39480044],
    [0.32194567, 0.14602909, 0.39362106],
    [0.31844477, 0.14497857, 0.39244549],
    [0.31494974, 0.14391333, 0.39127626],
    [0.31146605, 0.14282918, 0.39011024],
    [0.30798857, 0.1417297, 0.38895105],
    [0.30451661, 0.14061515, 0.38779953],
    [0.30105136, 0.13948445, 0.38665531],
    [0.2975886, 0.1383403, 0.38552159],
    [0.29408557, 0.13721193, 0.38442775],
]

_cmap = colors.ListedColormap(_flare_lut, "flare")
mpl.colormaps.register(_cmap, name="flare")
