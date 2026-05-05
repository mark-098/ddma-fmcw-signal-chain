"""
This code demonstrates the basic processing for a DDMA (Doppler-Division Multiple Access) radar chain for FMCW radars.

The data loader is designed to load the data streamed by a DCA1000EVM when capturing ADC data from an AWR2944 device.

This processing chain performs the following:
    1.) Data parsing
    2.) Interference mitigation
    3.) Computation of a range-Doppler map (RDM), as a non-coherent sum over all receivers and all DDMA sub-bands
    4.) Displays the results for a given frame and visualizes the effect of interference mitigation
    5.) Runs two consecutive CFAR (CA or CASO) searches across the Doppler and range domains.
        Each search direction can be configured and disabled independently
    6.) Performs DBSCAN clustering on the CFAR detection
    7.) Visualizes CFAR detections overlaid on the computed RDM and displays internal decision making of
        the CFAR algorithm, showcasing how close each point on the RDM has gotten to being a detection
    8.) Runs a KF based tracker in the RDM domain
    9.) Consolidates the track into a single synthetic track
    10.) Extracts micro-Doppler features for the synthetic track

Importantly, currently this script does not perform DDMA velocity disambiguation, nor does it build a DDMA MIMO
virtual array

Copyright (c) 2026 Mark Passia
"""
import pickle
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.ndimage import binary_dilation

from scipy.signal.windows import blackmanharris, gaussian, hann

#Project scope imports
from ddma_plotting import *
from ddma_dataclasses import *
from ddma_rdm_tracker import *
from ddma_data_loader import load_dca1000_real



#======================================================================================
#============================<<< Interference mitigation >>>===========================
#======================================================================================
def report_interference_stats(pr: ProcessingResults):
    corrupted_samples = pr.mtgBinaryMaskDilated.sum()
    corrupted_samples_percent = pr.mtgBinaryMaskDilated.mean() * 100.00
    print(f"Interference mitigation removed {corrupted_samples} samples. "
          f"\nAccounting for {corrupted_samples_percent:.2f} % of all samples within the frame.")

    # Report which chirp was worst, later this can be plotted for illustrative purposes
    removed_per_chirp_per_tx = pr.mtgBinaryMaskDilated.sum(axis=2)
    flat_idx = np.argmax(removed_per_chirp_per_tx)
    pr.worst_chirp_idx, pr.worst_rx_idx = np.unravel_index(flat_idx, removed_per_chirp_per_tx.shape)


def dilate_mask_1d(mask_bool, n_dilation):
    #Dilate a boolean mask along the last axis by n_dilation elements on each side
    structure = np.ones((1,) * (mask_bool.ndim - 1) + (2 * n_dilation + 1,),dtype=bool)
    return binary_dilation(mask_bool, structure=structure)


def hampel_mad_filter(frame, ps: ProcessingSettings):
    # 1.) Detect and flag outliers
    # Based on https://www.mathworks.com/help/dsp/ref/hampelfilter.html
    # Return a boolean array for the frame after running a per chirp outlier detector
    mag = np.abs(frame)
    med = np.median(mag, axis=-1, keepdims=True)
    mad = np.median(np.abs(mag - med), axis=-1, keepdims=True)
    thr = med + ps.interference_mitigation_cfg["k_mad"] * 1.4826 * mad
    mask_bad = mag > thr

    # 2.) Detect and flag adc maxing out
    mask_bad |= ps.interference_mitigation_cfg["adc_saturation"] < mag

    # 3.) Dilate the binary mask to prevent energy from the vicinity of corrupted samples messing up power spectrum
    mask_bad_dilated = dilate_mask_1d(mask_bad, ps.interference_mitigation_cfg["n_dilation"])

    # 3.) Return the flag
    return mask_bad, mask_bad_dilated


def interference_mitigation_main(frame, ps: ProcessingSettings, pr: ProcessingResults, verbose=True):
    chirps, rx, adc = frame.shape
    frame_imtg = np.asarray(frame, dtype=np.float32).copy()

    # 1.) Compute one DC per RX, average over all chirps and adc samples
    frame_imtg -= frame_imtg.mean(axis=(0, 2), keepdims=True)
    # 2.) Outlier flagging with MAD (Median Absolute Deviation) based hemp filter
    pr.mtgBinaryMask, pr.mtgBinaryMaskDilated = hampel_mad_filter(frame_imtg, ps)

    # 3.) Suppress the bad samples
    pr.mtgBinaryKeepMaskDilated = ~pr.mtgBinaryMaskDilated
    frame_imtg = frame_imtg * pr.mtgBinaryKeepMaskDilated

    # 4.) Report is asked
    if verbose:
        report_interference_stats(pr)

    return frame_imtg


#======================================================================================
#================================<<< CFAR Detector >>>=================================
#======================================================================================
def multi_mode_1D_CFAR(x, mode='CA', n_training=10, n_guard=3, p_fa=0.001):
    assert mode in ('CA', 'CASO'), "Parameter mode must be 'CA' or 'CASO'"

    if mode == 'CA': N_ref = 2*n_training
    elif mode=='CASO': N_ref = n_training

    a = N_ref * (p_fa**(-(1/N_ref)) - 1)

    N = len(x)
    noise, threshold = np.zeros(N), np.zeros(N)
    detections = np.zeros(N, dtype='bool')

    n_subWindow = n_training + n_guard
    x_padded = np.pad(x, (n_subWindow, n_subWindow), mode='symmetric')

    cumsum_0 = np.zeros(len(x_padded) + 1)
    cumsum_0[1:] = np.cumsum(x_padded)

    for i, cut in enumerate(x):
        i_padded = i + n_subWindow

        left_start = i_padded - (n_training + n_guard)
        left_end = i_padded - n_guard - 1

        right_start = i_padded + n_guard + 1
        right_end = i_padded + n_guard + n_training

        sum_left = cumsum_0[left_end + 1] - cumsum_0[left_start]
        sum_right = cumsum_0[right_end + 1] - cumsum_0[right_start]

        if mode=='CA':
            noise[i] = (sum_left + sum_right) / N_ref
        elif mode=='CASO':
            noise_left = sum_left / n_training
            noise_right = sum_right / n_training

            noise[i] = min(noise_left, noise_right)

        threshold[i] = a * noise[i]
        detections[i] = (cut > threshold[i])

    return noise, detections, threshold


# Run two orthogonal CFAR searches, get targets upon double detection.
def run_cfar_on_rd_map(input_rdm, cs: CaptureSettings, pr: ProcessingResults, ps: ProcessingSettings):
    N_samples, N_chirps = pr.rdmHeatmapMtg.shape

    enable_dd, enable_rd = ps.cfar_cfg['enable_dd'], ps.cfar_cfg['enable_rd']
    assert enable_dd or enable_rd, "At least one CFAR direction must be enabled"

    # If a pass is disabled it stays as all-True for the AND, and noise/threshold stay zero
    doppler_dim_detections = np.ones_like(input_rdm, dtype=bool)
    range_dim_detections = np.ones_like(input_rdm, dtype=bool)

    doppler_noise = np.zeros_like(input_rdm, dtype=float)
    range_noise = np.zeros_like(input_rdm, dtype=float)
    doppler_threshold = np.zeros_like(input_rdm, dtype=float)
    range_threshold = np.zeros_like(input_rdm, dtype=float)

    # Run CA-CFAR in the doppler domain (DD)
    if enable_dd:
        for range_bin in range(N_samples):
            doppler_vector = input_rdm[range_bin, :]
            noise, detections, threshold = multi_mode_1D_CFAR(doppler_vector,
                                                              mode=ps.cfar_cfg['dd_mode'],
                                                              n_training=ps.cfar_cfg['n_train_dd'],
                                                              n_guard=ps.cfar_cfg['n_guard_dd'],
                                                              p_fa=ps.cfar_cfg['p_fa_dd'])

            doppler_noise[range_bin, :] = noise
            doppler_threshold[range_bin, :] = threshold
            doppler_dim_detections[range_bin, :] = detections

    # Run CASO-CFAR in the range-domain (RD)
    if enable_rd:
        for doppler_bin in range(N_chirps):
            range_vector = input_rdm[:, doppler_bin]
            noise, detections, threshold = multi_mode_1D_CFAR(range_vector,
                                                  mode=ps.cfar_cfg['rd_mode'],
                                                  n_training=ps.cfar_cfg['n_train_rd'],
                                                  n_guard=ps.cfar_cfg['n_guard_rd'],
                                                  p_fa=ps.cfar_cfg['p_fa_rd'])

            range_noise[:, doppler_bin] = noise
            range_threshold[:, doppler_bin] = threshold
            range_dim_detections[:, doppler_bin] = detections

    # Check for matching detections and then obtain coordinates of detected targets
    detection_mask = doppler_dim_detections & range_dim_detections
    targets_coordinates = np.where(detection_mask)

    # Get coordinates of detected targets (row, column)
    pr.cfarResults["detection_mask"] = detection_mask
    pr.cfarResults["detection_doppler_mask"] = doppler_dim_detections
    pr.cfarResults["detection_range_mask"] = range_dim_detections
    pr.cfarResults["targets_coordinates"] = targets_coordinates
    pr.cfarResults["doppler_noise"] = doppler_noise
    pr.cfarResults["doppler_threshold"] = doppler_threshold
    pr.cfarResults["range_noise"] = range_noise
    pr.cfarResults["range_threshold"] = range_threshold


#======================================================================================
#=======================<<< Clustering (Post CFAR Processing) >>>======================
#======================================================================================
def report_cfar_and_cluster():
    pass

def prepare_target_coordinates(pr: ProcessingResults):
    targets_coordinates = pr.cfarResults["targets_coordinates"]
    if targets_coordinates[0].size == 0:
        return np.empty((0, 2), dtype=int)

    # Note: dbscan_input[i] = [range_bin, vel_bin]
    dbscan_input = np.column_stack(targets_coordinates)
    #print(dbscan_input[0])
    return dbscan_input


def dbscan_clustering(target_coordinates, epsilon=3, minimum_samples=3):
    # Expects input target_coordinates to be a (N, 2) array of (range_bin, doppler_bin)
    clusters = []
    centroids = []

    db = DBSCAN(eps=epsilon, min_samples=minimum_samples)
    db.fit(target_coordinates)
    labels = db.labels_
    unique_labels = set(labels) - {-1}

    # Obtain all the points in the current cluster and compute the centroid of each cluster
    for label in unique_labels:
        cluster_points = target_coordinates[labels == label]
        centroid = cluster_points.mean(axis=0)
        centroid_int = np.round(centroid).astype(int)
        clusters.append((label, cluster_points, centroid_int))

        range_bin, vel_bin = centroid_int
        centroids.append([range_bin, vel_bin])

    cluster_centroids = np.array(centroids)

    return labels, clusters, cluster_centroids


def cluster_detections(ps: ProcessingSettings, pr: ProcessingResults, verbose=True):
    N_samples, N_chirps = pr.rdmHeatmapMtg.shape

    target_coordinates = prepare_target_coordinates(pr)

    if ps.dbscan_cfg['filter_stationary']:
        stationary_clutter = np.arange((N_chirps//2 - 1), (N_chirps//2 + 2))
        moving_mask = ~np.isin(target_coordinates[:, 1], stationary_clutter)
        target_coordinates = target_coordinates[moving_mask]
        pr.cfarResults['targets_coordinates'] = target_coordinates  # keep pr in sync

    if target_coordinates.shape[0] == 0:
        labels, clusters, centroids = None, None, None
    else:
        labels, clusters, centroids = dbscan_clustering(target_coordinates,
                                                        epsilon=ps.dbscan_cfg['dbscan_eps'],
                                                        minimum_samples=ps.dbscan_cfg['dbscan_min_samp'])
    if verbose:
        report_cfar_and_cluster()

    return labels, clusters, centroids



#======================================================================================
#================================<<< DDMA Processing >>>===============================
#======================================================================================
def generate_windows(window_type, axis, cs:CaptureSettings):
    assert axis in ("doppler", "range"), "The 'axis' argument in 'generate_windows' must be 'doppler' or 'range'."

    # Case 1.): Range windows, reshape to last dim
    if axis == 'range':
        win_size = cs.num_adc_samples

        if window_type == 'Hanning':
            window = np.hanning(win_size).reshape(1, 1, 1, -1)
        if window_type == 'BlackmanHarris':
            window = blackmanharris(win_size).reshape(1, 1, 1, -1)

    # Case 2.): Doppler windows, reshape to first dim
    if axis == 'doppler':
        win_size = cs.num_chirps_per_loop

        if window_type == 'Hanning':
            window = np.hanning(win_size).reshape(-1, 1, 1, 1)
        if window_type == 'BlackmanHarris':
            window = blackmanharris(win_size).reshape(-1, 1, 1, 1)

    return window



def compute_ddma_rdm(frame_data, cs: CaptureSettings, ps: ProcessingSettings):
    assert ps.rdm_cfg['window'] in ('Hanning', 'BlackmanHarris', 'None'), \
        "The 'window' setting in rdm_cfg must be 'Hanning', 'BlackmanHarris' or 'None'."

    #DDMA RDM - split the frame data by DDMA subband
    frame_de_interleaved = frame_data.reshape(cs.num_chirps_per_loop, cs.num_chirp_loops, cs.num_rx, cs.num_adc_samples)

    range_win = np.ones(cs.num_adc_samples).reshape(1, 1, 1, -1)
    doppler_win = np.ones(cs.num_chirps_per_loop).reshape(-1, 1, 1, 1)

    # Obtain windows for further computations
    if ps.rdm_cfg['window'] != 'None':
        range_win = generate_windows(ps.rdm_cfg['window'], 'range', cs)
        doppler_win = generate_windows(ps.rdm_cfg['window'], 'doppler', cs)

    # Compute the windowed FFTs
    range_fft = np.fft.rfft(frame_de_interleaved * range_win, axis=3)
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft * doppler_win, axis=0), axes=0)

    #Perform non-coherent summation across all sub-bands and all receivers
    rdm_nonCoherent_linear = np.sum(np.abs(doppler_fft)**2, axis=(1, 2))
    rdm_nonCoherent_dB = 10 * np.log10(np.abs(rdm_nonCoherent_linear) + 1e-12)

    return rdm_nonCoherent_linear.T, rdm_nonCoherent_dB.T



#===================================================================================================
#====================================<<< Slow-time processing >>>===================================
#===================================================================================================
# Computes the un-windowed range FFT once, used for slow-time / micro-Doppler analysis
def compute_post_rfft_slow_time(frame_data, cs: CaptureSettings):
    # Shape: (CHIRPS_PER_LOOP, CHIRP_LOOPS, RX, NUM_ADC_POST_FFT)
    frame_de_interleaved = frame_data.reshape(cs.num_chirps_per_loop, cs.num_chirp_loops, cs.num_rx, cs.num_adc_samples)
    return np.fft.rfft(frame_de_interleaved, axis=3)

# Used for per frame processing
# Picks the DBSCAN cluster with the most points. Returns its centroid (rboi, dboi) or None
def pick_largest_cluster(pr: ProcessingResults):
    if pr.dbscan_clusters is None or len(pr.dbscan_clusters) == 0:
        return None

    # dbscan_clusters is a list of (label, cluster_points, centroid_int)
    largest = max(pr.dbscan_clusters, key=lambda c: len(c[1]))
    centroid_int = largest[2]
    rboi, dboi = int(centroid_int[0]), int(centroid_int[1])
    return rboi, dboi

# Generates a 1D window for STFT use according to ps.stft_cfg
def generate_stft_window(ps: ProcessingSettings):
    win_type = ps.stft_cfg['window']
    win_len = ps.stft_cfg['winLen']

    if win_type == 'Gaussian':
        window = gaussian(win_len, std=ps.stft_cfg['winStd'])
    elif win_type == 'Hanning':
        window = hann(win_len)
    elif win_type == 'BlackmanHarris':
        window = blackmanharris(win_len)
    else:
        raise ValueError(f"Unknown STFT window type: '{win_type}'")

    return window


# Computes the stft for a single frame at a target (range bin of interest, doppler bin of interest)
# Operates on a single chosen DDMA subband, non-coherently summed across all receivers
def compute_stft_ddma(rboi, dboi, stft_subband, cs: CaptureSettings, ps: ProcessingSettings, pr: ProcessingResults):
    # 1.) Pick the slow-time vector at the requested range bin and subband
    # postRfftSlowTime shape: (CHIRPS_PER_LOOP, CHIRP_LOOPS, RX, NUM_ADC_POST_FFT)
    slow_time = pr.postRfftSlowTime[:, stft_subband, :, rboi]   # (CHIRPS_PER_LOOP, RX)

    # 2.) STFT parameters
    hop = ps.stft_cfg['hop']
    mfft = ps.stft_cfg['mfft']
    win_len = ps.stft_cfg['winLen']
    window = generate_stft_window(ps)

    n_samples = slow_time.shape[0]
    n_frames = (n_samples - win_len) // hop + 1

    # 3.) Compute STFT per receiver, then non-coherently sum
    stft_per_rx = np.zeros((cs.num_rx, mfft, n_frames), dtype=float)
    for rx in range(cs.num_rx):
        rx_signal = slow_time[:, rx]

        for frame_idx in range(n_frames):
            start = frame_idx * hop
            segment = rx_signal[start:start + win_len] * window
            spectrum = np.fft.fftshift(np.fft.fft(segment, n=mfft))
            stft_per_rx[rx, :, frame_idx] = np.abs(spectrum) ** 2

    stft_linear = np.sum(stft_per_rx, axis=0)   # (mfft, n_frames)
    stft_dB = 10 * np.log10(stft_linear + 1e-12)

    # 4.) Build physical axes for plotting
    # Each STFT frame represents `hop` chirps, and each chirp lasts cs.frame_period / num_chirps_per_loop seconds
    chirp_period = cs.frame_period / cs.num_chirps_per_loop
    stft_time_axis = np.arange(n_frames) * hop * chirp_period

    # Doppler axis: same vel_resolution scaled by the FFT length ratio
    doppler_axis = (np.arange(mfft) - mfft // 2) * cs.vel_resolution * (cs.num_chirps_per_loop / mfft)

    # 5.) Store into pr
    pr.stft_linear = stft_linear
    pr.stft_dB = stft_dB
    pr.stft_time_axis = stft_time_axis
    pr.stft_doppler_axis = doppler_axis
    pr.stft_rboi = rboi
    pr.stft_dboi = dboi

    return stft_linear, stft_dB



#Drives the stft computation per track
def compute_stft_driver():
    pass



#=======================================================================================================
#======================<<< Slow time processing per track (Feature extraction) >>>======================
#=======================================================================================================
# Walks a consolidated (or any) TrackHistory, recomputes the post-rFFT cube on the fly for each frame,
# computes the STFT at the track's measured (rboi, dboi)
def extract_stft_features(history: TrackHistory, raw_adc_data, cs: CaptureSettings, ps: ProcessingSettings, stft_subband = 0):
    # 1.) Collect frames where the track had a real measurement
    valid_snaps = [s for s in history.snapshots if s.measurement is not None]
    n_meas = len(valid_snaps)

    # 2.) Probe STFT shape with first frame to enable pre-allocation.
    # Temporary ProcessingResults for the per-frame work
    pr_tmp = ProcessingResults()

    first_snap = valid_snaps[0]
    first_frame_data = raw_adc_data[first_snap.frame_idx]
    pr_tmp.mtgRadarCube = interference_mitigation_main(first_frame_data, ps, pr_tmp, verbose=False)
    pr_tmp.postRfftSlowTime = compute_post_rfft_slow_time(pr_tmp.mtgRadarCube, cs)

    rboi0, dboi0 = int(first_snap.measurement[0]), int(first_snap.measurement[1])
    compute_stft_ddma(rboi0, dboi0, stft_subband, cs, ps, pr_tmp)

    mfft, n_stft_frames = pr_tmp.stft_dB.shape

    # 3.) Preallocate everything
    frame_indices = np.zeros(n_meas, dtype=int)
    range_bins = np.zeros(n_meas, dtype=int)
    doppler_bins  = np.zeros(n_meas, dtype=int)
    stft_dB = np.zeros((n_meas, mfft, n_stft_frames), dtype=float)
    stft_linear = np.zeros((n_meas, mfft, n_stft_frames), dtype=float)

    # First frame is already done, fill it in
    frame_indices[0] = first_snap.frame_idx
    range_bins[0] = rboi0
    doppler_bins[0] = dboi0
    stft_dB[0] = pr_tmp.stft_dB
    stft_linear[0] = pr_tmp.stft_linear

    # 4.) Walk remaining frames
    for i, snap in tqdm(enumerate(valid_snaps[1:], start=1),
                        total=len(valid_snaps) - 1,
                        desc=f"STFT features (track {history.track_id})",
                        unit="frame"):
        frame_data = raw_adc_data[snap.frame_idx]
        pr_tmp.mtgRadarCube = interference_mitigation_main(frame_data, ps, pr_tmp, verbose=False)
        pr_tmp.postRfftSlowTime = compute_post_rfft_slow_time(pr_tmp.mtgRadarCube, cs)

        rboi, dboi = int(snap.measurement[0]), int(snap.measurement[1])
        compute_stft_ddma(rboi, dboi, stft_subband, cs, ps, pr_tmp)

        frame_indices[i] = snap.frame_idx
        range_bins[i] = rboi
        doppler_bins[i] = dboi
        stft_dB[i] = pr_tmp.stft_dB
        stft_linear[i] = pr_tmp.stft_linear

    # 5.) Pack into a TrackFeatures and attach to the history
    features = TrackFeatures(track_id=history.track_id,
                             frame_indices=frame_indices,
                             range_bins=range_bins,
                             doppler_bins=doppler_bins,
                             stft_dB=stft_dB,
                             stft_linear=stft_linear,
                             stft_time_axis=pr_tmp.stft_time_axis,
                             stft_doppler_axis=pr_tmp.stft_doppler_axis,
                             stft_subband=stft_subband)
    history.features = features

    return features



#================================================================================================
#===========================<<< Main per frame processing functions >>>==========================
#================================================================================================
def process_single_frame(frame_data,
                         cs:CaptureSettings,
                         ps:ProcessingSettings,
                         pr:ProcessingResults,
                         visualize=False,
                         rboi=None, dboi=None,
                         stft_subband=0):

    # 0.) For multi-frame usage of this function - clear stale (conditional) data
    pr.stft_dB, pr.stft_linear, pr.stft_rboi, pr.stft_dboi = None, None, None, None

    # 1.) Store the raw frame and run interference mitigation
    pr.rawRadarCube = frame_data
    pr.mtgRadarCube = interference_mitigation_main(pr.rawRadarCube, ps, pr, verbose=visualize)

    # 2.) Compute DDMA RDM ()
    pr.rdmHeatmapMtg, pr.rdmHeatmapMtg_db  = compute_ddma_rdm(pr.mtgRadarCube, cs, ps)
    pr.rdmHeatmapRaw, pr.rdmHeatmapRaw_db  = compute_ddma_rdm(pr.rawRadarCube, cs, ps)

    # 3.) Un-windowed range FFT on the mitigated cube, kept for slow-time analysis
    pr.postRfftSlowTime = compute_post_rfft_slow_time(pr.mtgRadarCube, cs)

    # 4.) Display the results of interference mitigation
    if visualize:
        visualize_interference_mitigation_ddma_rdm(cs, ps, pr)

    # 5.) Run CFAR (by default pass the mitigated rdm)
    run_cfar_on_rd_map(pr.rdmHeatmapMtg, cs, pr, ps)

    # 6.) Perform clustering and other detection post-processing
    pr.dbscan_labels, pr.dbscan_clusters, pr.dbscan_centroids = cluster_detections(ps, pr)

    # 7.) Visualize cfar and clustering results
    if visualize:
        visualize_cfar_and_tracker_results(pr, cs, ps)

    # 8.) Slow-time STFT analysis. If rboi/dboi not supplied, fall back to largest cluster.
    if rboi is None or dboi is None:
        cluster_pick = pick_largest_cluster(pr)
        if cluster_pick is not None:
            rboi, dboi = cluster_pick

    if rboi is not None and dboi is not None:
        compute_stft_ddma(rboi, dboi, stft_subband, cs, ps, pr)
        if visualize:
            slow_time_analysis(cs, ps, pr)
    else:
        if visualize:
            print("No DBSCAN clusters in this frame, skipping slow-time analysis.")


# Runs process_single_frame on the requested frame. Split with process_single_frame made for trackers convince
def main_processing_single_frame(frame_ui, cs: CaptureSettings, ps: ProcessingSettings, pr: ProcessingResults, visualize=True):
    frame_data = pr.rawAdcData[frame_ui, :, :, :]
    process_single_frame(frame_data, cs, ps, pr, visualize=visualize)


#===================================================================================================
#=====================================<<< Tracker driver code >>>===================================
#===================================================================================================
def extract_measurements_for_tracker(pr: ProcessingResults):
    centroids = pr.dbscan_centroids
    if centroids is None or len(centroids) == 0:
        return np.empty((0, 2), dtype=float)

    # Tracker math is float-based, even though bins are integers
    return np.asarray(centroids, dtype=float)


# Per-frame logging, save to the dedicated data structure
def snapshot_track(track: Track, frame_idx, was_updated, measurement):
    return TrackSnapshot(frame_idx=frame_idx,
                         track_id=track.id,
                         was_updated=was_updated,
                         measurement=(None if measurement is None else np.asarray(measurement, dtype=float).copy()),
                         predicted_position=track.position.copy(),
                         kf_state=track.state.copy(),
                         age=track.age,
                         hits_total=track.hits_total,
                         consecutive_hits=track.consecutive_hits,
                         misses_total=track.misses_total,
                         misses_consecutive=track.misses_consecutive,
                         is_confirmed=track.is_confirmed)



# Walk the tracker's current state after process_frame(), write logging entries. Returns the set of live IDs this frame
def log_frame(tracker: Tracker, frame_idx, misses_total_before: dict, prev_live_ids: set, tr: TrackerResults):
    frame_snapshots = []
    current_live_ids = set()

    # Go over all tracks detected by the tracker during this frame
    for track in tracker.tracks:
        current_live_ids.add(track.id)

        # Decide whether this track was updated or missed in this frame.
        prev_misses = misses_total_before.get(track.id)
        # 1a.) Brand-new track
        if prev_misses is None:
            was_updated = True
            measurement = track.last_measurement
        # 1b.) Old track
        else:
            was_updated = (track.misses_total == prev_misses)
            measurement = track.last_measurement if was_updated else None

        snap = snapshot_track(track, frame_idx, was_updated, measurement)
        frame_snapshots.append(snap)

        # 2a.) Birth: first time with this ID, create a TrackHistory
        if track.id not in tr.track_histories:
            tr.track_histories[track.id] = TrackHistory(track_id=track.id, birth_frame=frame_idx,
                                                        birth_measurement=(None if measurement is None else np.asarray(measurement, dtype=float).copy()))

        # 3.) For all tracks append snapshot to the track's history
        history = tr.track_histories[track.id]
        history.snapshots.append(snap)
        if was_updated and measurement is not None:
            history.last_measurement = np.asarray(measurement, dtype=float).copy()

    # Set the list of frame_snapshots form the processed frame
    tr.per_frame_snapshots[frame_idx] = frame_snapshots

    # Detect deaths: IDs that were alive last frame but aren't anymore.
    # The death frame is the *previous* frame (last frame in which the track was actually present)
    died_ids = prev_live_ids - current_live_ids
    for dead_id in died_ids:
        history = tr.track_histories.get(dead_id)
        if history is not None and history.death_frame is None:
            last_seen = history.snapshots[-1].frame_idx if history.snapshots else frame_idx - 1
            history.finalize(death_frame=last_seen)

    return current_live_ids



def report_during_tracker_operations(tracker, frame_idx, num_frames, measurements):
    n_live = len(tracker.tracks)
    n_conf = sum(1 for t in tracker.tracks if t.is_confirmed)
    print(f"Tracker frame {frame_idx + 1}/{num_frames} meas={measurements.shape[0]:>3} "
          f"live={n_live:>3} confirmed={n_conf:>3} total_ids={tracker.next_track_id}")



def run_tracker_over_all_frames(raw_adc_data, cs: CaptureSettings, ps: ProcessingSettings,
                                pr: ProcessingResults, mcpi_ps: multiCpiProcessingSettings,
                                process_frame_fn, verbose = True, saveTrackerResults=False,
                                cache_rdms=False):

    # RDM cache for subsequent tracker visualization
    rdm_cache = [] if cache_rdms else None

    tracker = Tracker(dist_thresh=mcpi_ps.dist_thresh,
                      max_misses=mcpi_ps.max_misses,
                      min_hits_confirm=mcpi_ps.min_hits_confirm,
                      dt=mcpi_ps.dt,
                      process_noise=mcpi_ps.process_noise,
                      meas_noise=mcpi_ps.meas_noise)

    tr = TrackerResults()
    num_frames = raw_adc_data.shape[0]
    tr.num_frames_processed = num_frames

    prev_live_ids = set()

    pbar = tqdm(range(num_frames), desc="Tracker", unit="frame")
    for frame_idx in pbar:
        # 1.) Run the full per-frame processing chain on this frame
        frame_data = raw_adc_data[frame_idx]
        process_frame_fn(frame_data, cs, ps, pr, visualize=False)

        if cache_rdms:
            rdm_cache.append(pr.rdmHeatmapMtg_db.copy())

        # 2.) Adapt DBSCAN centroids to tracker input measurements
        measurements = extract_measurements_for_tracker(pr)
        tr.measurements_per_frame[frame_idx] = measurements.shape[0]

        # 3.) Snapshot misses_total for every live track BEFORE process_frame, to later infer updated vs. missed.
        misses_total_before = {t.id: t.misses_total for t in tracker.tracks}

        # 4.) Push measurements through the tracker
        tracker.process_frame(measurements)

        # 5.) Log everything
        prev_live_ids = log_frame(tracker, frame_idx, misses_total_before, prev_live_ids, tr)

        # 6.) Update tdqm
        n_live = len(tracker.tracks)
        n_conf = sum(1 for t in tracker.tracks if t.is_confirmed)
        pbar.set_postfix(meas=measurements.shape[0], live=n_live, confirmed=n_conf, total_ids=tracker.next_track_id)

    # Finalize any tracks that were still alive when the run ended.
    last_frame = num_frames - 1
    for history in tr.track_histories.values():
        if history.death_frame is None:
            last_seen = history.snapshots[-1].frame_idx if history.snapshots else last_frame
            history.finalize(death_frame=last_seen)

    # Print short report on tracker performance
    if verbose:
        n_total = len(tr.track_histories)
        n_confirmed = len(tr.confirmed_track_ids())
        print(f"Tracker done. {num_frames} frames, {n_total} total tracks, {n_confirmed} ever-confirmed")

    # Save tracker results for further processing
    if saveTrackerResults:
        save_path = cs.raw_file.replace('.bin', '_tracker_results.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(tr, f)
        print(f"Tracker results saved to {save_path}")

    if cache_rdms:
        return tr, rdm_cache

    return tr, None



def load_tracks(file_loc):
    with open(file_loc, 'rb') as f:
        tr = pickle.load(f)
    print("File loaded successfully!")
    return tr



# Keep only track with more than n frames of lifetime
def keep_long_tracks(tr: TrackerResults, min_lifetime = 10) -> TrackerResults:
    filtered = TrackerResults()

    # 1.) Go over all tracks and check for their lifetime
    for tid, h in tr.track_histories.items():
        if h.lifetime_frames > min_lifetime:
            # a.) Copy the track history data over all frames that it existed
            filtered.track_histories[tid] = h

            # b.) Copy the tracks snapshot data for the current frame, iterate over all snapshots/frames
            for snap in h.snapshots:
                # Initialize the filtered snapshot array (True on the first iteration of the loop)
                if snap.frame_idx not in filtered.per_frame_snapshots:
                    filtered.per_frame_snapshots[snap.frame_idx] = []

                # After creating the filtered snapshot array append snaps from the copied track
                filtered.per_frame_snapshots[snap.frame_idx].append(snap)

    return filtered



# Consolidate the obtained tracks into a single synthetic track for the whole file
# Use only when sure that during data acquisition a SINGLE meaningful target was present
def consolidate_filtered_tracks(tr: TrackerResults) -> TrackHistory:
    histories = list(tr.track_histories.values())

    # 1.) Find frames with overlapping tracks
    frame_owners = {}
    for h in histories:
        for snap in h.snapshots:
            frame_owners.setdefault(snap.frame_idx, []).append(h.track_id)

    overlapping_frames = {f: ids for f, ids in frame_owners.items() if len(ids) > 1}

    if overlapping_frames:
        print(f"WARNING: {len(overlapping_frames)} frames have multiple tracks alive and will be SKIPPED during consolidation.")
        print(f"\tSkipped frames -> track ids:")
        for f in sorted(overlapping_frames.keys()):
            print(f"\t\tframe {f}: tracks {overlapping_frames[f]}")

    skipped_set = set(overlapping_frames.keys())

    # 2.) Concatenate all snapshots except those in skipped frames
    all_snapshots = []
    for h in histories:
        for snap in h.snapshots:
            if snap.frame_idx not in skipped_set:
                all_snapshots.append(snap)

    all_snapshots.sort(key=lambda s: s.frame_idx)

    # 3.) Build the synthetic TrackHistory
    birth_frame = all_snapshots[0].frame_idx
    death_frame = all_snapshots[-1].frame_idx

    birth_meas = None
    for s in all_snapshots:
        if s.measurement is not None:
            birth_meas = s.measurement.copy()
            break

    last_meas = None
    for s in reversed(all_snapshots):
        if s.measurement is not None:
            last_meas = s.measurement.copy()
            break

    consolidated = TrackHistory(track_id=-1,
                                birth_frame=birth_frame,
                                birth_measurement=birth_meas,
                                last_measurement=last_meas,
                                snapshots=all_snapshots)
    consolidated.finalize(death_frame=death_frame)

    print(f"Consolidated {len(histories)} tracks into one synthetic track: "
          f"frames {birth_frame}..{death_frame}, "
          f"{len(all_snapshots)} snapshots, "
          f"{len(skipped_set)} frames skipped (overlap), "
          f"gaps={consolidated.lifetime_frames - len(all_snapshots)}")

    return consolidated



def track_post_processing_main(tr: TrackerResults, raw_adc_data, cs: CaptureSettings, ps: ProcessingSettings,
                               min_lifetime=10, stft_subband=0, generate_movie=False):
    # Clean up of the tracks, keep only tracks with above 10 frames of lifetime
    tr = keep_long_tracks(tr, min_lifetime=min_lifetime)
    total_tracked_frames = sum(h.lifetime_frames for h in tr.track_histories.values())
    print(f"Total number of tracked frames for track with a lifetime above {min_lifetime} frames: {total_tracked_frames}")

    # Info for manual file inspection, longest track stats
    tr_sorted = sorted(tr.track_histories.values(), key=lambda h: h.lifetime_frames, reverse=True)
    max_track_frames, max_track_id = tr_sorted[0].lifetime_frames, tr_sorted[0].track_id
    print(f"The track with the most frames is tid {max_track_id}, with {max_track_frames} frames")

    # 3.) Consolidate filtered tracks into a single synthetic track
    tr_consolidated = consolidate_filtered_tracks(tr)

    # 4.) Extract STFT features on the consolidated track (recomputes per frame on the fly)
    extract_stft_features(tr_consolidated, raw_adc_data, cs, ps, stft_subband=stft_subband)

    return tr_consolidated




#===================================================================================================
#=====================================<<< TF Dataset creation >>>===================================
#===================================================================================================




#======================================================================================
#===========================<<< Setup and initialization >>>===========================
#======================================================================================
if __name__ == "__main__":
    # Set the capture variables for loading and processing, initialize the script
    DATA_PATH = r"adc_data_Raw_0_angled.bin"

    # Data acquisition settings
    NUM_RX = 4 # Number of active receivers
    NUM_CHIRPS = int(6 * 64)  # chirps per frame (all TX sub-bands combined)
    NUM_CHIRPS_PER_LOOP = 64
    NUM_CHIRP_LOOPS = 6 #Dictates the number of sub-bands
    NUM_ADC_SAMPLES = 192  # fast-time samples per chirp

    RANGE_RES = 0.78125000
    VEL_RES = 0.10680109

    FRAME_PERIOD = 0.125 # 125ms <-> 8Hz
    FRAME_TO_PROCESS = 628

    # Program flow control
    PROCESS_SINGLE_FRAME = True
    EXTRACT_TRACKS_FROM_FILE = True
    CREATE_TRACKER_MOVIE = True

    SAVE_PROCESSED_TRACKS = True
    LOAD_CACHED_TRACKS = True

    # Initialize both capture and processing setting classes
    cs = CaptureSettings(raw_file=DATA_PATH,
                         num_rx=NUM_RX,
                         num_adc_samples=NUM_ADC_SAMPLES,
                         num_chirps_per_loop=NUM_CHIRPS_PER_LOOP,
                         num_chirp_loops=NUM_CHIRP_LOOPS,
                         range_resolution=RANGE_RES,
                         vel_resolution=VEL_RES,
                         frame_period=FRAME_PERIOD)

    mcpi_ps = multiCpiProcessingSettings(dt=cs.frame_period)

    ps = ProcessingSettings()
    pr = ProcessingResults()

    # Load the raw adc data from the .bin file just once. With shape: (FRAME, CHIRPS, RX, ADC)
    raw_adc_data = load_dca1000_real(cs.raw_file, cs.num_rx, cs.num_total_chirps(), cs.num_adc_samples)
    pr.rawAdcData = raw_adc_data

    # ----------------------<<< BASIC PROGRAM FLOW CONTROL >>>----------------------
    # Single frame inspection with interactive plots
    if PROCESS_SINGLE_FRAME:
        main_processing_single_frame(FRAME_TO_PROCESS, cs, ps, pr, visualize=True)

    # Execute the tracker, go over the whole file
    if EXTRACT_TRACKS_FROM_FILE:
        tr, rdm_cache = run_tracker_over_all_frames(raw_adc_data, cs, ps, pr,
                                                    mcpi_ps, process_single_frame,
                                                    verbose=True,
                                                    saveTrackerResults=SAVE_PROCESSED_TRACKS,
                                                    cache_rdms=True)
        tr_consolidated = track_post_processing_main(tr, raw_adc_data, cs, ps)

    # Development purposes: load the track data
    if LOAD_CACHED_TRACKS and not EXTRACT_TRACKS_FROM_FILE:
        tr = load_tracks("adc_data_Raw_0_angled_tracker_results.pkl")
        tr_consolidated = track_post_processing_main(tr)

    if CREATE_TRACKER_MOVIE:
        tracker_movie(tr_consolidated, cs, rdm_cache, output_path="tracker_movie.mp4", fps=8)


