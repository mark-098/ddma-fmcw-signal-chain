import numpy as np
#============================================================================================================
#--------------------------------------<<< DCA1000EVM Data Loading >>>--------------------------------------
#============================================================================================================
# Reads and reshapes raw ADC data from a DCA1000EVM capture file. For: Real + 4 Lane LVDS
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