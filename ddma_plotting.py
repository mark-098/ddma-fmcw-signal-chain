import numpy as np
import matplotlib.pyplot as plt

from ddma_dataclasses import *
from matplotlib.patches import Circle

#======================================================================================
#=============================<<< Visualization utils >>>==============================
#======================================================================================
def rdm_axes(cs: CaptureSettings, vel_tick_step=1.0, range_tick_step=10.0):
    num_range_bins = cs.num_adc_samples // 2 + 1
    num_vel_bins = cs.num_chirps_per_loop

    range_axis_m = np.arange(num_range_bins) * cs.range_resolution
    vel_axis_mps = (np.arange(num_vel_bins) - num_vel_bins // 2) * cs.vel_resolution

    extent = [vel_axis_mps[0],
              vel_axis_mps[-1],
              range_axis_m[0],
              range_axis_m[-1]]

    # Symmetric velocity ticks around 0
    vmax = np.max(np.abs(vel_axis_mps))
    vel_ticks = np.arange(-np.floor(vmax), np.floor(vmax) + 1e-12, vel_tick_step)

    # Range ticks
    range_ticks = np.arange(0, range_axis_m[-1], range_tick_step)

    return extent, range_axis_m, vel_axis_mps, vel_ticks, range_ticks


def visualize_interference_mitigation_ddma_rdm(cs:CaptureSettings, ps:ProcessingSettings, pr:ProcessingResults):
    extent, range_axis_m, vel_axis_mps, vel_ticks, range_ticks = rdm_axes(cs)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # ==============<<< 1.) RDM obtained using interference mitigated data >>>==============
    ax0 = ax[0, 0]
    im0 = ax0.imshow(pr.rdmHeatmapMtg_db, origin='lower', aspect='auto', extent=extent)
    ax0.set_title('Mitigated DDMA RDM')
    ax0.set_xlabel("Velocity [m/s]")
    ax0.set_ylabel("Range [m]")
    ax0.set_xticks(vel_ticks)
    ax0.set_yticks(range_ticks)

    # ==============<<< 2.) RDM obtained using raw adc data >>>==============
    ax1 = ax[0, 1]
    im1 = ax1.imshow(pr.rdmHeatmapRaw_db, origin='lower', aspect='auto')
    ax1.set_title('Raw data DDMA RDM')
    ax1.set_xlabel("Velocity [m/s]")
    ax1.set_ylabel("Range [m]")
    ax0.set_xticks(vel_ticks)
    ax0.set_yticks(range_ticks)

    # ==============<<< 3.) The worst quality chirp of the frame after interference mitigation >>>==============
    ax2 = ax[1, 0]
    ax2.plot(pr.mtgRadarCube[pr.worst_chirp_idx, pr.worst_rx_idx, :])
    ax2.set_xlabel("ADC sample")
    ax2.set_ylabel("ADC value (post DC rem)")
    ax2.set_title("Worst chirp in frame - hampel filtered")
    ax2.grid()

    # ==============<<< 4.) The worst quality chirp of the frame before interference mitigation >>>==============
    ax3 = ax[1, 1]
    ax3.plot(pr.rawRadarCube[pr.worst_chirp_idx, pr.worst_rx_idx, :])
    ax3.set_xlabel("ADC sample")
    ax3.set_ylabel("ADC value")
    ax3.set_title("Worst chirp in frame - raw samples")
    ax3.grid()

    # Set overall plot layout
    plt.suptitle("Comparison: DDMA RDM hampel MAD (Median Average Deviation) filter vs. raw data")
    plt.tight_layout()
    plt.show()


def visualize_cfar_and_tracker_results(pr: ProcessingResults, cs: CaptureSettings, ps: ProcessingSettings):
    extent, range_axis_m, vel_axis_mps, vel_ticks, range_ticks = rdm_axes(cs)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # ==============<<< 1.) RDM with overlaid cfar detection (red) and dbscan clusters (white) >>>==============
    ax0 = ax[0, 0]

    im0 = ax0.imshow(pr.rdmHeatmapMtg_db, origin='lower', aspect='auto', extent=extent)
    for range_bin, doppler_bin in zip(pr.cfarResults['targets_coordinates'][0], pr.cfarResults['targets_coordinates'][1]):
        ax0.scatter(vel_axis_mps[doppler_bin], range_axis_m[range_bin], s=40, facecolors='none', edgecolors='red', linewidths=1.5)

    for (range_bin, doppler_bin) in pr.dbscan_centroids:
        ax0.scatter(vel_axis_mps[doppler_bin], range_axis_m[range_bin], s=40, facecolors='none', edgecolors='white', linewidths=1.5)

    ax0.set_title("CFAR detections (red) and DBSCAN clusters (white)")
    ax0.set_xlabel("Velocity [m/s]")
    ax0.set_ylabel("Range [m]")
    ax0.set_xticks(vel_ticks)
    ax0.set_yticks(range_ticks)

    # ==============<<< 2.) CFAR doppler domain detection mask >>>==============
    ax1 = ax[0, 1]
    im1 = ax1.imshow(pr.cfarResults["detection_doppler_mask"], origin='lower', aspect='auto', extent=extent, cmap='gray_r')

    ax1.set_title("CFAR doppler domain detection mask")
    ax1.set_xlabel("Velocity [m/s]")
    ax1.set_ylabel("Range [m]")
    ax1.set_xticks(vel_ticks)
    ax1.set_yticks(range_ticks)

    # ==============<<< 3.) CFAR range domain detection mask >>>==============
    ax2 = ax[1, 0]
    im2 = ax2.imshow(pr.cfarResults["detection_range_mask"], origin='lower', aspect='auto', extent=extent, cmap='gray_r')

    ax2.set_title("CFAR range domain detection mask")
    ax2.set_xlabel("Velocity [m/s]")
    ax2.set_ylabel("Range [m]")
    ax2.set_xticks(vel_ticks)
    ax2.set_yticks(range_ticks)

    # ==============<<< 4.) CFAR detector internal snr/threshold values >>>==============
    # How close each cell came to being a detection
    qs = []
    if ps.cfar_cfg['enable_dd']:
        qs.append(pr.rdmHeatmapMtg / np.maximum(pr.cfarResults["doppler_threshold"], 1e-12))
    if ps.cfar_cfg['enable_rd']:
        qs.append(pr.rdmHeatmapMtg / np.maximum(pr.cfarResults["range_threshold"], 1e-12))

    q_min = np.minimum.reduce(qs)  # min over only the enabled passes
    q_db = 10 * np.log10(q_min + 1e-12)

    ax3 = ax[1, 1]
    v_lim = 10  # ±10 dB around the decision boundary
    im3 = ax3.imshow(q_db, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-v_lim, vmax=v_lim, extent=extent)
    ax3.contour(q_db, levels=[0], colors='black', linewidths=0.8, extent=extent)
    fig.colorbar(im3, ax=ax3, label="signal/threshold [dB]")

    ax3.set_title("CFAR margin (limiting pass) - 0 dB = on threshold")
    ax3.set_xlabel("Velocity [m/s]")
    ax3.set_ylabel("Range [m]")
    ax3.set_xticks(vel_ticks)
    ax3.set_yticks(range_ticks)

    plt.suptitle(f"Visualization of CFAR (Range: {ps.cfar_cfg['rd_mode']}, Doppler: {ps.cfar_cfg['dd_mode']}) detection performance")
    plt.tight_layout()
    plt.show()



















