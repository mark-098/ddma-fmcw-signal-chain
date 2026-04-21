"""
This code demonstrates the basic processing for a DDMA (Doppler-Division Multiple Access) radar chain for FMCW radars.

The data loader is designed to load the data streamed by a DCA1000EVM when capturing ADC data from an AWR2944 device.

This processing chain performs the following:
    1.) Data parsing
    2.) Interference mitigation
    3.) Computation of a range-Doppler map (RDM), as a non-coherent sum over all receivers and all DDMA sub-bands
    4.) Displays the results for a given frame and visualizes the effect of interference mitigation

Importantly, currently this script does not perform DDMA velocity disambiguation, nor does it build a DDMA MIMO
virtual array

Copyright (c) 2026 Mark Passia
"""
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from scipy.ndimage import binary_dilation


#============================================================================================================
#--------------------------------------<<< DCA1000EVM Data Loading >>>--------------------------------------
#============================================================================================================
# Reads and reshapes raw ADC data from a DCA1000EVM capture file.
def load_dca1000_real(file_path, num_rx, num_chirps, num_adc_samples):
    samples_to_parse = 0
    raw_data = np.fromfile(file_path, dtype=np.int16)

    # 1.) Compute how manny frames can be parsed - warn the user if not all data will get used up
    samples = raw_data.size
    samples_per_frame = num_chirps * num_rx * num_adc_samples
    modulo_remainder = samples % samples_per_frame

    if modulo_remainder != 0:
        samples_to_parse = samples - modulo_remainder
        print("Warning: File size is not a multiple of the frame size. Check num_rx/num_chirps/num_adc_samples.")
    else:
        samples_to_parse = samples

    data_to_parse = raw_data[:samples_to_parse]
    num_frames = int(samples_to_parse / samples_per_frame)

    # 2.) Parse the raw ADC data and swap the last two axes to get (num_frames, num_chirps, num_rx, num_adc_samples)
    output_array = data_to_parse.reshape((num_frames, num_chirps, num_adc_samples, num_rx))
    output_array = output_array.swapaxes(2, 3)

    print(f"File loaded successfully. With {num_frames} frames.")

    return output_array


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
#================================<<< DDMA Processing >>>===============================
#======================================================================================
def compute_ddma_rdm(frame_data, cs: CaptureSettings, ps: ProcessingSettings):
    #DDMA RDM - split the frame data by DDMA subband
    frame_de_interleaved = frame_data.reshape(cs.num_chirps_per_loop, cs.num_chirp_loops, cs.num_rx, cs.num_adc_samples)
    range_fft = np.fft.rfft(frame_de_interleaved, axis=3)
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
    doppler_noncoherent = np.sum(np.abs(doppler_fft)**2, axis=(1, 2))

    rdm = 10 * np.log10(np.abs(doppler_noncoherent) + 1e-12)

    return rdm.T

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

    #
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax0 = ax[0, 0]
    im0 = ax0.imshow(pr.rdmHeatmapMtg, origin='lower', aspect='auto', extent=extent)
    ax0.set_title('Mitigated DDMA RDM')
    ax0.set_xlabel("Velocity [m/s]")
    ax0.set_ylabel("Range [m]")
    ax0.set_xticks(vel_ticks)
    ax0.set_yticks(range_ticks)

    ax1 = ax[0, 1]
    im1 = ax1.imshow(pr.rdmHeatmapRaw, origin='lower', aspect='auto')
    ax1.set_title('Raw data DDMA RDM')
    ax1.set_xlabel("Velocity [m/s]")
    ax1.set_ylabel("Range [m]")
    ax0.set_xticks(vel_ticks)
    ax0.set_yticks(range_ticks)

    ax2 = ax[1, 0]
    ax2.plot(pr.mtgRadarCube[pr.worst_chirp_idx, pr.worst_rx_idx, :], color='blue', linestyle='--')
    #ax2.plot(pr.rawRadarCube[pr.worst_chirp_idx, pr.worst_rx_idx, :], color='red', linestyle=':')
    ax2.set_xlabel("ADC sample")
    ax2.set_ylabel("ADC value (post DC rem)")
    ax2.set_title("Worst chirp in frame - hampel filtered")
    ax2.grid()

    ax3 = ax[1, 1]
    ax3.plot(pr.rawRadarCube[pr.worst_chirp_idx, pr.worst_rx_idx, :])
    ax3.set_xlabel("ADC sample")
    ax3.set_ylabel("ADC value")
    ax3.set_title("Worst chirp in frame - raw samples")
    ax3.grid()


    plt.suptitle("Comparison: DDMA RDM hampel MAD (Median Average Deviation) filter vs. raw data")
    plt.tight_layout()
    plt.show()


#======================================================================================
#===========================<<< Main processing function >>>===========================
#======================================================================================
def main_processing_frame(frame_ui, cs:CaptureSettings, ps:ProcessingSettings, pr:ProcessingResults):
    # 1.) Load the raw radar cube. Shape: (FRAME, CHIRPS, RX, ADC)
    pr.rawAdcData = load_dca1000_real(cs.raw_file, cs.num_rx, cs.num_total_chirps(), cs.num_adc_samples)

    # 2.) Select the frame under investigation and perform interference mitigation
    pr.rawRadarCube = pr.rawAdcData[frame_ui, :, :, :]
    pr.mtgRadarCube = interference_mitigation_main(pr.rawRadarCube, ps, pr)

    # 3.) Compute DDMA RDM
    pr.rdmHeatmapMtg = compute_ddma_rdm(pr.mtgRadarCube, cs, ps)
    pr.rdmHeatmapRaw = compute_ddma_rdm(pr.rawRadarCube, cs, ps)


    # 4.) Display the processing results
    visualize_interference_mitigation_ddma_rdm(cs, ps, pr)


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
