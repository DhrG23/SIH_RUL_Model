"""
Standard health estimation: conventional time-domain + frequency-domain
feature extraction, and a Mahalanobis-distance health index fit on
known-healthy baseline data. This is the textbook condition-based-
monitoring approach (not novel, per your call to use "the standard method"
here and save the custom architecture for RUL/fault detection).

Hardened for dynamic (aviation-style) operation:
  - GaussianMixtureHealthEstimator: regime-conditioned health index, so a
    throttle change doesn't register as mechanical degradation.
  - order_track_resample(): time-domain -> angle-domain resampling using a
    tachometer/RPM signal, so spectral features stay stationary under
    engine acceleration.
  - StandardFeatureExtractor.extract(..., rpm_signal=...): optional order-
    tracking path; extract_stft(): STFT-based transient energy features
    when a tachometer isn't available.
"""
import numpy as np
from scipy.signal import stft
from sklearn.mixture import GaussianMixture
from .interfaces import FeatureExtractor, HealthEstimator


def order_track_resample(signal_window: np.ndarray, rpm_signal: np.ndarray, sample_rate: float,
                          orders_per_rev: int = 64) -> np.ndarray:
    """Resample a time-domain signal into the angular domain using an
    instantaneous RPM trace of the same length. Output has a fixed number
    of samples per shaft revolution, so a spectrum taken on the result
    (an "order spectrum") stays stationary regardless of engine speed
    changes during the window -- this is what makes order tracking correct
    where a plain rFFT on the raw time signal is not, under acceleration.
    """
    rpm = np.asarray(rpm_signal, dtype=float)
    t = np.arange(len(signal_window)) / sample_rate
    rev_per_sec = rpm / 60.0
    # cumulative shaft angle (in revolutions) via trapezoidal integration of speed
    theta = np.concatenate([[0.0], np.cumsum((rev_per_sec[:-1] + rev_per_sec[1:]) / 2.0 * np.diff(t))])
    total_revs = theta[-1]
    if total_revs <= 0:
        return np.asarray(signal_window, dtype=float)  # not rotating -- nothing to order-track
    n_out = max(int(total_revs * orders_per_rev), 8)
    theta_uniform = np.linspace(0, total_revs, n_out)
    return np.interp(theta_uniform, theta, signal_window)


class StandardFeatureExtractor(FeatureExtractor):
    """Time-domain + frequency-domain features on one signal window.
    Works for any vibration/current/acoustic-style 1-D condition-monitoring
    signal; swap in domain-specific features (e.g. bearing fault
    frequencies) if you have a known fault mechanism."""

    _NAMES = ["rms", "peak", "crest_factor", "kurtosis", "skewness", "std",
              "peak_to_peak", "spectral_centroid", "spectral_energy_hf_ratio",
              "dominant_freq", "stft_transient_energy"]

    def feature_names(self):
        return list(self._NAMES)

    def extract(self, signal_window: np.ndarray, sample_rate: float, rpm_signal=None,
                orders_per_rev: int = 64) -> dict:
        x = np.asarray(signal_window, dtype=float)
        if rpm_signal is not None:
            # Order tracking: resample to angle domain BEFORE any spectral
            # feature, so speed changes within the window don't smear the
            # spectrum. Time-domain stats (rms, kurtosis, ...) are computed
            # on the original signal -- those don't need order tracking.
            x_spec = order_track_resample(x, rpm_signal, sample_rate, orders_per_rev=orders_per_rev)
            # FIX: eff_fs must be samples-PER-REVOLUTION (orders_per_rev),
            # not samples-per-second -- x_spec is already in the angle
            # domain (fixed samples/revolution), so rfftfreq's bin units
            # come out as CYCLES-PER-REVOLUTION ("orders"), which is the
            # whole point of order tracking. The old
            # `len(x_spec) / (len(x) / sample_rate)` computed a samples-
            # per-second-shaped number instead, so "dominant_freq" wasn't
            # actually in order units at all under a changing RPM.
            eff_fs = float(orders_per_rev)
        else:
            x_spec = x
            eff_fs = sample_rate

        n = len(x)
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        crest = peak / (rms + 1e-9)
        mean, std = x.mean(), x.std() + 1e-9
        kurt = np.mean((x - mean) ** 4) / (std ** 4) - 3.0
        skew = np.mean((x - mean) ** 3) / (std ** 3)
        ptp = x.max() - x.min()

        # --- frequency / order domain ---
        freqs = np.fft.rfftfreq(len(x_spec), d=1.0 / eff_fs)
        mag = np.abs(np.fft.rfft(x_spec))
        total_energy = mag.sum() + 1e-9
        spectral_centroid = float((freqs * mag).sum() / total_energy)
        hf_cutoff = freqs.max() / 2.0
        hf_energy_ratio = float(mag[freqs > hf_cutoff].sum() / total_energy)
        dominant_freq = float(freqs[np.argmax(mag)]) if len(mag) else 0.0

        # --- STFT transient-energy feature: catches short-lived spikes a
        # single whole-window FFT averages away. Peak-band-energy variance
        # across time segments = how "bursty" the high-frequency content is.
        stft_energy = self._stft_transient_energy(x, sample_rate)

        return {
            "rms": rms, "peak": peak, "crest_factor": crest, "kurtosis": kurt,
            "skewness": skew, "std": std, "peak_to_peak": ptp,
            "spectral_centroid": spectral_centroid,
            "spectral_energy_hf_ratio": hf_energy_ratio,
            "dominant_freq": dominant_freq,
            "stft_transient_energy": stft_energy,
        }

    def _stft_transient_energy(self, x, sample_rate, nperseg=None):
        nperseg = nperseg or max(16, len(x) // 8)
        if len(x) < 2 * nperseg:
            return 0.0
        _, _, Zxx = stft(x, fs=sample_rate, nperseg=nperseg)
        band_energy_per_frame = np.sum(np.abs(Zxx) ** 2, axis=0)  # energy per time segment
        return float(np.std(band_energy_per_frame) / (np.mean(band_energy_per_frame) + 1e-9))

    def extract_vector(self, signal_window, sample_rate, rpm_signal=None, orders_per_rev: int = 64):
        d = self.extract(signal_window, sample_rate, rpm_signal=rpm_signal, orders_per_rev=orders_per_rev)
        return np.array([d[name] for name in self._NAMES])


class MahalanobisHealthEstimator(HealthEstimator):
    """Standard multivariate health index: Mahalanobis distance of the
    current feature vector from the healthy-baseline distribution. This IS
    the conventional method used in ISO 13374-style condition monitoring
    (equivalent to Hotelling's T^2 statistic).

    NOTE: assumes ONE healthy regime. Use GaussianMixtureHealthEstimator
    below if the asset has distinct healthy operating regimes (idle /
    climb / cruise, etc.) -- otherwise regime changes will falsely score
    as degradation."""

    def __init__(self, reg=1e-6):
        self.reg = reg

    def fit(self, healthy_features: np.ndarray):
        X = np.asarray(healthy_features, dtype=float)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = cov + self.reg * np.eye(cov.shape[0])  # regularize for invertibility
        self.inv_cov_ = np.linalg.inv(cov)
        return self

    def health_index(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        diff = X - self.mean_
        # Mahalanobis distance, vectorized: sqrt(diff @ inv_cov @ diff.T) diag
        m = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff))
        return m if X.shape[0] > 1 else m[0]


class GaussianMixtureHealthEstimator(HealthEstimator):
    """Regime-conditioned health index. Fits a GMM on healthy baseline data
    -- each Gaussian component represents one operating regime (idle,
    climb, cruise, ...), discovered unsupervised, not hand-labeled. Health
    index = Mahalanobis distance to the NEAREST regime's mean/covariance,
    so a throttle change that moves the asset from one legitimate healthy
    regime to another does not register as degradation -- only genuine
    departure from every known-healthy regime does."""

    def __init__(self, n_regimes=3, reg_covar=1e-3, covariance_type="diag", seed=0):
        # FIX: reg_covar raised from 1e-6 -> 1e-3 default. The old value let
        # a GMM component collapse onto a tight little cluster in limited
        # healthy data and score it as "very healthy", while genuinely
        # typical points scored WORSE -- which is why the ablation test
        # showed GMM's max distance coming out higher than plain
        # Mahalanobis (backwards from the claim). covariance_type="diag"
        # is also now the default: "full" needs enough samples per regime
        # to estimate a full covariance matrix reliably, which small
        # healthy-only training sets often don't have.
        self.n_regimes, self.reg_covar = n_regimes, reg_covar
        self.covariance_type, self.seed = covariance_type, seed

    @staticmethod
    def select_n_regimes(healthy_features, max_regimes=5, covariance_type="diag",
                          reg_covar=1e-3, seed=0):
        """Pick n_regimes by BIC instead of guessing. Fits 1..max_regimes
        components and returns the count with the lowest BIC (the
        standard penalized-likelihood way to choose GMM component count --
        rewards fit, penalizes needless extra regimes)."""
        X = np.asarray(healthy_features, dtype=float)
        best_n, best_bic = 1, np.inf
        for n in range(1, max_regimes + 1):
            gmm = GaussianMixture(n_components=n, covariance_type=covariance_type,
                                   reg_covar=reg_covar, random_state=seed).fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_n, best_bic = n, bic
        return best_n

    def fit(self, healthy_features: np.ndarray):
        X = np.asarray(healthy_features, dtype=float)
        self.gmm_ = GaussianMixture(n_components=self.n_regimes, covariance_type=self.covariance_type,
                                     reg_covar=self.reg_covar, random_state=self.seed)
        self.gmm_.fit(X)
        if self.covariance_type == "diag":
            # diag covariances_ come back as (n_regimes, n_features) -- expand
            # to full diagonal matrices so the rest of this class (which
            # assumes a full covariance matrix per regime) doesn't change.
            self.inv_covs_ = [np.diag(1.0 / v) for v in self.gmm_.covariances_]
        else:
            self.inv_covs_ = [np.linalg.inv(c) for c in self.gmm_.covariances_]
        return self

    def health_index(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        n_regimes = self.gmm_.means_.shape[0]
        dists = np.zeros((X.shape[0], n_regimes))
        for r in range(n_regimes):
            diff = X - self.gmm_.means_[r]
            dists[:, r] = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_covs_[r], diff))
        m = dists.min(axis=1)  # distance to NEAREST regime, not a single global mean
        return m if X.shape[0] > 1 else m[0]

    def regime_of(self, features: np.ndarray):
        """Which regime each reading was closest to -- useful for logging/
        debugging, not required by the HealthEstimator interface."""
        X = np.atleast_2d(np.asarray(features, dtype=float))
        return self.gmm_.predict(X)

