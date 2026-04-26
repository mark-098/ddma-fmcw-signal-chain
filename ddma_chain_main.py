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

Importantly, currently this script does not perform DDMA velocity disambiguation, nor does it build a DDMA MIMO
virtual array

Copyright (c) 2026 Mark Passia
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.ndimage import binary_dilation

#Project scope imports
from ddma_plotting import *
from ddma_dataclasses import *
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
# TODO: Toggle enable/disable for range or doppler cfar
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

    # Remove zero Doppler points - works only for stationary radar
    if ps.dbscan_cfg['filter_stationary']:
        stationary_clutter = np.arange((N_chirps//2 - 1), (N_chirps//2 + 2))
        moving_mask = ~np.isin(target_coordinates[:, 1], stationary_clutter)
        target_filtered = target_coordinates[moving_mask]
        target_coordinates = target_filtered

        #Overwrite the target coordinates with the filtered results
        pr.cfarResults['targets_coordinates'] = target_coordinates

    target_coordinates = prepare_target_coordinates(pr)

    if target_coordinates.shape[0] == 0:
        labels, clusters, centroids = None, None, None
    else:
        labels, clusters, centroids = dbscan_clustering(target_coordinates,
                                                        epsilon=ps.dbscan_cfg['dbscan_eps'],
                                                        minimum_samples=ps.dbscan_cfg['dbscan_min_samp'])

    if verbose:
        report_cfar_and_cluster()

    # Note: labels = value for each pair in target_coordinates, -1 if not a cluster, else cluster ID
    #       clusters = list of point pairs within each cluster
    #       centroids = point pairs of cluster centroids
    return labels, clusters, centroids



#======================================================================================
#================================<<< DDMA Processing >>>===============================
#======================================================================================
def compute_ddma_rdm(frame_data, cs: CaptureSettings, ps: ProcessingSettings):
    #DDMA RDM - split the frame data by DDMA subband
    frame_de_interleaved = frame_data.reshape(cs.num_chirps_per_loop, cs.num_chirp_loops, cs.num_rx, cs.num_adc_samples)
    range_fft = np.fft.rfft(frame_de_interleaved, axis=3)
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

    #Perform non-coherent summation across all sub-bands and all receivers
    rdm_nonCoherent_linear = np.sum(np.abs(doppler_fft)**2, axis=(1, 2))
    rdm_nonCoherent_dB = 10 * np.log10(np.abs(rdm_nonCoherent_linear) + 1e-12)

    return rdm_nonCoherent_linear.T, rdm_nonCoherent_dB.T



#======================================================================================
#===========================<<< Main processing function >>>===========================
#======================================================================================
def main_processing_frame(frame_ui, cs:CaptureSettings, ps:ProcessingSettings, pr:ProcessingResults):
    # 1.) Load the raw radar cube. Shape: (FRAME, CHIRPS, RX, ADC)
    pr.rawAdcData = load_dca1000_real(cs.raw_file, cs.num_rx, cs.num_total_chirps(), cs.num_adc_samples)

    # 2.) Select the frame under investigation and perform interference mitigation
    pr.rawRadarCube = pr.rawAdcData[frame_ui, :, :, :]
    pr.mtgRadarCube = interference_mitigation_main(pr.rawRadarCube, ps, pr)

    # 3.) Compute DDMA RDM ()
    pr.rdmHeatmapMtg, pr.rdmHeatmapMtg_db = compute_ddma_rdm(pr.mtgRadarCube, cs, ps)
    pr.rdmHeatmapRaw, pr.rdmHeatmapRaw_db = compute_ddma_rdm(pr.rawRadarCube, cs, ps)

    # 4.) Display the results of interference mitigation
    #visualize_interference_mitigation_ddma_rdm(cs, ps, pr)

    # 5.) Run CFAR (by default pass the mitigated rdm)
    run_cfar_on_rd_map(pr.rdmHeatmapMtg, cs, pr, ps)

    # 6.) Perform clustering and other detection post-processing
    pr.dbscan_labels, pr.dbscan_clusters, pr.dbscan_centroids = cluster_detections(ps, pr)
    #labels, clusters, centroids = cluster_detections(ps, pr)

    # 7.) Visualize cfar and clustering results
    visualize_cfar_and_tracker_results(pr, cs, ps)


#======================================================================================
#===========================<<< Setup and initialization >>>===========================
#======================================================================================
if __name__ == "__main__":
    # Set the capture variables for loading and processing, initialize the script
    DATA_PATH = r"adc_data_Raw_0_angled.bin"

    NUM_RX = 4 # Number of active receivers
    NUM_CHIRPS = int(6 * 64)  # chirps per frame (all TX sub-bands combined)
    NUM_CHIRPS_PER_LOOP = 64
    NUM_CHIRP_LOOPS = 6
    NUM_ADC_SAMPLES = 192  # fast-time samples per chirp

    RANGE_RES = 0.78125000
    VEL_RES = 0.10680109

    FRAME_TO_PROCESS = 160

    # Initialize both capture and processing setting classes
    capture_settings = CaptureSettings(raw_file=DATA_PATH,
                                       num_rx=NUM_RX,
                                       num_adc_samples=NUM_ADC_SAMPLES,
                                       num_chirps_per_loop=NUM_CHIRPS_PER_LOOP,
                                       num_chirp_loops=NUM_CHIRP_LOOPS,
                                       range_resolution=RANGE_RES,
                                       vel_resolution=VEL_RES)
    processing_settings = ProcessingSettings()
    processing_results = ProcessingResults()

    # Execute main processing function
    main_processing_frame(FRAME_TO_PROCESS, capture_settings, processing_settings, processing_results)
