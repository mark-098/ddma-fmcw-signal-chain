import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field

#======================================================================================
#=======================<<< Placeholders for processing data >>>=======================
#======================================================================================
@dataclass
# Capture settings - abbrev. as 'cs' in further code
class CaptureSettings:
    """ Describes settings used during data capture """
    raw_file: str
    num_rx: int
    num_adc_samples: int
    num_chirps_per_loop: int
    num_chirp_loops: int

    range_resolution: float
    vel_resolution: float

    frame_period: float

    def num_total_chirps(self) -> int:
        return self.num_chirp_loops * self.num_chirps_per_loop

    def num_subbands(self) -> int:
        return self.num_chirp_loops

    def num_adc_post_fft(self) -> int:
        return self.num_adc_samples // 2


@dataclass
# Processing settings - abbrev. as 'ps' in further code
class ProcessingSettings:
    """ Settings used during the processing """
    interference_mitigation_cfg: dict = field(default_factory=lambda:
                                        {'k_mad': 5.0, #Lower value, more samples zeroed
                                         'n_dilation': 5, #Higher value, more samples zeroed
                                         'adc_saturation': 32000.0}
                                        )

    rdm_cfg: dict = field(default_factory=lambda:
                    {'window':'None'} #Supported windows: BlackmanHarris, Hamming, None
                    )

    cfar_cfg: dict = field(default_factory=lambda:
                     {'dd_mode': 'CASO', #Cfar mode in the doppler-domain (dd)
                      'rd_mode': 'CA', #Cfar mode in the range-domain (rd)
                      'n_train_dd': 5, #Training samples dd
                      'n_train_rd': 5, #Training samples rd
                      'n_guard_dd': 2,  #Guard cells dd
                      'n_guard_rd': 3,  #Guard cells rd
                      'p_fa_dd': 0.01, #Probability of false alarm dd
                      'p_fa_rd': 0.01, #Probability of false alarm rd
                      'enable_dd': True,
                      'enable_rd': True})

    dbscan_cfg: dict = field(default_factory=lambda:
                       {'dbscan_eps': 2,
                        'dbscan_min_samp': 2,
                        'filter_stationary': True})

    stft_cfg: dict = field(default_factory=lambda :
                     {'hop': 1,
                      'mfft': 256,
                      'winLen': 8,
                      'winStd': 5,
                      'window': 'Gaussian'}) # Supported: Gaussian, Hanning, BlackmanHarris


@dataclass
# Processing results - abbrev. as 'pr' in further code
class ProcessingResults:
    """ Results of the processing: final and intermediate """
    # Loaded ADC data, (radar cube for all frames)
    rawAdcData: np.ndarray = None

    # Radar cubes, raw and filtered (for current frame)
    rawRadarCube: np.ndarray = None
    mtgRadarCube: np.ndarray = None

    # Mitigation results, raw and dilated
    mtgBinaryMask: np.ndarray = None
    mtgBinaryMaskDilated: np.ndarray = None
    mtgBinaryKeepMaskDilated: np.ndarray = None

    worst_chirp_idx: int = None
    worst_rx_idx: int = None

    # Post range-FFT slow time - un-windowed
    postRfftSlowTime: np.ndarray = None

    # STFT results, computed at a (rboi, dboi)
    stft_dB: np.ndarray = None
    stft_linear: np.ndarray = None
    stft_time_axis: np.ndarray = None
    stft_doppler_axis: np.ndarray = None
    stft_rboi: int = None
    stft_dboi: int = None


    # Rdm heatmap
    rdmHeatmapMtg: np.ndarray = None
    rdmHeatmapRaw: np.ndarray = None

    rdmHeatmapMtg_db: np.ndarray = None
    rdmHeatmapRaw_db: np.ndarray = None

    # Cfar results
    cfarResults : dict = field(default_factory=lambda:
                            {'detection_mask': None,
                             'detection_doppler_mask': None,
                             'detection_range_mask': None,
                             'targets_coordinates': None,
                             'doppler_noise': None,
                             'doppler_threshold': None,
                             'range_noise': None,
                             'range_threshold': None,
                             'dbscan_input': None})

    # Clustering results
    dbscan_labels: np.ndarray = None
    dbscan_clusters: np.ndarray = None
    dbscan_centroids: np.ndarray = None


@dataclass
# Multiple CPI processing settings - abbrev. as 'mcpi_ps' in further code
class multiCpiProcessingSettings:
    # Tracker settings
    dt: float
    process_noise: list = field(default_factory=lambda: [0.5, 0.5])
    meas_noise: list = field(default_factory=lambda: [1.0, 1.0])
    dist_thresh: float = 10.0
    max_misses: int = 0
    min_hits_confirm: int = 3

    # Extraction mode
    extract_noise: bool = False


#======================================================================================
#=============================<<< Tracker logging types >>>============================
#======================================================================================
@dataclass
# Snapshot of one track's state in a single frame. Captures both what the KF believed vs the measurement (if any)
class TrackSnapshot:
    frame_idx: int
    track_id: int
    was_updated: bool

    # Raw measurement [range_bin, doppler_bin] from DBSCAN, or None on a miss
    measurement: Optional[np.ndarray]
    predicted_position: np.ndarray
    kf_state: np.ndarray

    # Track-level counters at this point in time
    age: int
    hits_total: int
    consecutive_hits: int
    misses_total: int
    misses_consecutive: int
    is_confirmed: bool



@dataclass
# Lifetime record of one track. Built up incrementally and finalized when the track is deleted (or when the file ends).
class TrackHistory:
    track_id: int
    birth_frame: int
    death_frame: Optional[int] = None

    # First and last raw measurements
    birth_measurement: Optional[np.ndarray] = None
    last_measurement: Optional[np.ndarray] = None

    # Per-frame snapshots over the track's life (in chronological order)
    snapshots: List[TrackSnapshot] = field(default_factory=list)

    # Summary stats - filled in at finalization
    lifetime_frames: Optional[int] = None
    peak_consecutive_hits: int = 0
    total_hits: int = 0
    total_misses: int = 0
    ever_confirmed: bool = False
    confirmed_at_frame: Optional[int] = None

    # Called when the track disappears (deleted or run ends)
    def finalize(self, death_frame: int):
        self.death_frame = death_frame
        self.lifetime_frames = death_frame - self.birth_frame + 1

        if self.snapshots:
            self.peak_consecutive_hits = max(s.consecutive_hits for s in self.snapshots)
            self.total_hits = max(s.hits_total for s in self.snapshots)
            self.total_misses = max(s.misses_total for s in self.snapshots)
            self.ever_confirmed = any(s.is_confirmed for s in self.snapshots)
            for s in self.snapshots:
                if s.is_confirmed:
                    self.confirmed_at_frame = s.frame_idx
                    break

    # Optional per-track feature data, populated by extract_stft_features()
    features: Optional["TrackFeatures"] = None


@dataclass
# Top-level container for everything the tracker produced over a full file. Two views:
# per_frame_snapshots[frame_idx] - list of TrackSnapshot for that frame
# track_histories[track_id] - full lifetime TrackHistory for that track
class TrackerResults:
    per_frame_snapshots: Dict[int, List[TrackSnapshot]] = field(default_factory=dict)
    track_histories: Dict[int, TrackHistory] = field(default_factory=dict)

    # Per-frame measurement counts (input to the tracker), useful for diagnostics
    measurements_per_frame: Dict[int, int] = field(default_factory=dict)

    # Total number of frames processed
    num_frames_processed: int = 0

    def all_track_ids(self) -> List[int]:
        return sorted(self.track_histories.keys())

    def confirmed_track_ids(self) -> List[int]:
        return sorted(tid for tid, h in self.track_histories.items() if h.ever_confirmed)




@dataclass
# Per-track classifier feature data. One TrackFeatures instance per TrackHistory.
# Keep this flat - frame-aligned arrays indexed by position.
class TrackFeatures:
    track_id: int

    # Frame-aligned arrays (length = number of frames where the track had a measurement)
    frame_indices: np.ndarray = None    # (N_meas,) int
    range_bins:    np.ndarray = None    # (N_meas,) int
    doppler_bins:  np.ndarray = None    # (N_meas,) int

    # Per-frame STFT stack
    stft_dB:     np.ndarray = None      # (N_meas, mfft, n_stft_frames) float
    stft_linear: np.ndarray = None      # (N_meas, mfft, n_stft_frames) float

    # STFT axes - same for every frame, stored once
    stft_time_axis:    np.ndarray = None    # (n_stft_frames,)
    stft_doppler_axis: np.ndarray = None    # (mfft,)

    stft_subband: int = 0