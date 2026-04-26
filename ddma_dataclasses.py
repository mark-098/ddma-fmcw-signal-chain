import numpy as np
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
                    {'window':'Han'}
                    )

    cfar_cfg: dict = field(default_factory=lambda:
                     {'dd_mode': 'CA', #Cfar mode in the doppler-domain (dd)
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
                        'filter_stationary': False})


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



