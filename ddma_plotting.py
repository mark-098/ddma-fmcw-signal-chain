import os
import shutil
import numpy as np
import imageio.v2 as imageio

import matplotlib.pyplot as plt

from ddma_dataclasses import *
from matplotlib.patches import Circle
from tqdm import tqdm

#=============================================================================================
#===========================<<< Visualization helper functions >>>============================
#=============================================================================================
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



#=============================================================================================
#=============================<<< Static visualization utils >>>==============================
#=============================================================================================
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



def slow_time_analysis(cs: CaptureSettings, ps: ProcessingSettings, pr: ProcessingResults):
    extent, range_axis_m, vel_axis_mps, vel_ticks, range_ticks = rdm_axes(cs)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # ==============<<< 1.) RDM with crosshairs at chosen cluster >>>==============
    ax0 = ax[0]
    ax0.imshow(pr.rdmHeatmapMtg_db, origin='lower', aspect='auto', extent=extent)

    r_m = pr.stft_rboi * cs.range_resolution
    v_mps = (pr.stft_dboi - cs.num_chirps_per_loop // 2) * cs.vel_resolution

    ax0.axhline(y=r_m, color='white', linestyle='--', linewidth=1.2)
    ax0.axvline(x=v_mps, color='white', linestyle='--', linewidth=1.2)
    ax0.scatter(v_mps, r_m, s=40, facecolors='none', edgecolors='red', linewidths=1.5)

    ax0.set_title(f"RDM with selected cluster (rboi={pr.stft_rboi}, dboi={pr.stft_dboi})")
    ax0.set_xlabel("Velocity [m/s]")
    ax0.set_ylabel("Range [m]")
    ax0.set_xticks(vel_ticks)
    ax0.set_yticks(range_ticks)

    # ==============<<< 2.) STFT spectrogram at rboi >>>==============
    ax1 = ax[1]
    stft_extent = [pr.stft_time_axis[0], pr.stft_time_axis[-1],
                   pr.stft_doppler_axis[0], pr.stft_doppler_axis[-1]]
    im1 = ax1.imshow(pr.stft_dB, origin='lower', aspect='auto', extent=stft_extent)
    fig.colorbar(im1, ax=ax1, label="STFT [dB]")

    ax1.set_title(f"STFT spectrogram at rboi={pr.stft_rboi} ({r_m:.2f} m)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity [m/s]")

    plt.suptitle("Slow-time analysis")
    plt.tight_layout()
    plt.show()



#==============================================================================================
#=============================<<< Dynamic visualization utils >>>==============================
#==============================================================================================
def tracker_movie(consolidated: TrackHistory, cs: CaptureSettings, rdm_cache,
                  output_path = "tracker_movie.mp4", fps = 8, trail_length = 10, dpi = 100):

    assert consolidated.features is not None, \
        "Consolidated track has no features. Call extract_stft_features() first."

    num_frames = len(rdm_cache)
    snaps_by_frame = {s.frame_idx: s for s in consolidated.snapshots}

    # Build a frame_idx to position-in-features-array map for fast STFT lookup
    feat = consolidated.features
    feat_idx_by_frame = {int(f): i for i, f in enumerate(feat.frame_indices)}

    extent, range_axis_m, vel_axis_mps, vel_ticks, range_ticks = rdm_axes(cs)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 1.) Global colour scales (no per-frame flicker)
    rdm_stacked = np.stack(rdm_cache)
    rdm_db_min = float(np.percentile(rdm_stacked,  5))
    rdm_db_max = float(np.percentile(rdm_stacked, 99))

    stft_db_min = float(np.percentile(feat.stft_dB,  5))
    stft_db_max = float(np.percentile(feat.stft_dB, 99))

    # 2.) Build the figure once (two panels, like slow_time_analysis)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax_rdm, ax_stft = ax[0], ax[1]

    rdm_im = ax_rdm.imshow(rdm_cache[0], origin='lower', aspect='auto', extent=extent, vmin=rdm_db_min, vmax=rdm_db_max)
    fig.colorbar(rdm_im, ax=ax_rdm, label='RDM [dB]')

    crosshair_h = ax_rdm.axhline(y=range_axis_m[0], color='white', linestyle='--', linewidth=1.2)
    crosshair_v = ax_rdm.axvline(x=vel_axis_mps[0],  color='white', linestyle='--', linewidth=1.2)

    rdm_title = ax_rdm.set_title("")
    ax_rdm.set_xlabel("Velocity [m/s]")
    ax_rdm.set_ylabel("Range [m]")
    ax_rdm.set_xticks(vel_ticks)
    ax_rdm.set_yticks(range_ticks)

    stft_extent = [feat.stft_time_axis[0], feat.stft_time_axis[-1], feat.stft_doppler_axis[0], feat.stft_doppler_axis[-1]]
    stft_blank = np.full_like(feat.stft_dB[0], stft_db_min)
    stft_im = ax_stft.imshow(stft_blank, origin='lower', aspect='auto', extent=stft_extent, vmin=stft_db_min, vmax=stft_db_max)
    fig.colorbar(stft_im, ax=ax_stft, label='STFT [dB]')

    stft_title = ax_stft.set_title("")
    ax_stft.set_xlabel("Time [s]")
    ax_stft.set_ylabel("Velocity [m/s]")

    # 3.) Stream frames out via imageio
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1)

    try:
        for frame_idx in tqdm(range(num_frames), desc="Rendering movie", unit="frame"):
            update_movie_two_panel_helper(frame_idx=frame_idx,
                                          num_frames=num_frames,
                                          rdm_db=rdm_cache[frame_idx],
                                          rdm_im=rdm_im,
                                          rdm_title=rdm_title,
                                          crosshair_h=crosshair_h,
                                          crosshair_v=crosshair_v,
                                          stft_im=stft_im,
                                          stft_title=stft_title,
                                          stft_blank=stft_blank,
                                          snaps_by_frame=snaps_by_frame,
                                          feat=feat,
                                          feat_idx_by_frame=feat_idx_by_frame,
                                          cs=cs)

            fig.canvas.draw()
            frame_img = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(frame_img)
    finally:
        writer.close()
        plt.close(fig)

    print(f"Movie saved: {output_path}")


# Update the persistent two-panel figure (RDM + STFT) for one movie frame
def update_movie_two_panel_helper(frame_idx, num_frames, rdm_db, rdm_im, rdm_title, crosshair_h, crosshair_v,
                                  stft_im, stft_title, stft_blank, snaps_by_frame, feat, feat_idx_by_frame, cs):
    # ---- Left panel: RDM background ----
    rdm_im.set_data(rdm_db)
    snap = snaps_by_frame.get(frame_idx)

    if snap is None:
        # No track on this frame - hide crosshairs, blank the STFT panel
        crosshair_h.set_ydata([np.nan, np.nan])
        crosshair_v.set_xdata([np.nan, np.nan])
        rdm_title.set_text(f"Frame {frame_idx}/{num_frames - 1} [no track]")
        stft_im.set_data(stft_blank)
        stft_title.set_text("STFT - no track this frame")
        return

    # Snapshot present - place crosshairs (measurement if present, else prediction)
    if snap.measurement is not None:
        r_bin, d_bin = snap.measurement[0], snap.measurement[1]
        tag = " [tracking]"
    else:
        r_bin, d_bin = snap.predicted_position[0], snap.predicted_position[1]
        tag = " [coast]"

    r_m   = r_bin * cs.range_resolution
    v_mps = (d_bin - cs.num_chirps_per_loop // 2) * cs.vel_resolution
    crosshair_h.set_ydata([r_m, r_m])
    crosshair_v.set_xdata([v_mps, v_mps])
    rdm_title.set_text(f"Frame {frame_idx}/{num_frames - 1}{tag}")

    # ---- Right panel: STFT for this frame (only if a measurement exists) ----
    feat_pos = feat_idx_by_frame.get(frame_idx)
    if feat_pos is None:
        # Coast frame - no STFT was computed (extract_stft_features only runs on measured frames)
        stft_im.set_data(stft_blank)
        stft_title.set_text("STFT - coast frame, no measurement")
    else:
        stft_frame = feat.stft_dB[feat_pos]
        stft_im.set_data(stft_frame)
        stft_im.set_clim(vmin=stft_frame.min(), vmax=stft_frame.max())
        stft_title.set_text(f"STFT at rboi={int(feat.range_bins[feat_pos])} ({r_m:.2f} m)")








