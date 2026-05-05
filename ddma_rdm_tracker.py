# Intended file: rd_tracker.py (REPLACE all content)
# This file contains the Kalman Filter, Track, and Tracker classes.
# This version is updated to explicitly track consecutive/total hits & misses.

import numpy as np
from scipy.optimize import linear_sum_assignment


#===================================================================================================
#=========================<<< Kalman filter math and core structure class >>>=======================
#===================================================================================================
class KalmanFilter_CA_RD:
    """
    A Kalman Filter for a 2D (Range, Doppler) Constant Acceleration model.
    The state is [r, r_dot, r_ddot, d, d_dot, d_ddot].
    The measurement is [r, d].
    """

    def __init__(self, dt, process_noise, meas_noise):
        self.dt = dt

        # State vector [r, r_dot, r_ddot, d, d_dot, d_ddot]
        self.x = np.zeros((6, 1))
        # State covariance
        self.P = np.eye(6) * 500.0

        # State Transition Matrix (F)
        F_ca = np.array([
            [1, dt, 0.5 * dt ** 2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        self.F = np.block([
            [F_ca, np.zeros((3, 3))],
            [np.zeros((3, 3)), F_ca]
        ])

        # Measurement Matrix (H)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        # Process Noise Covariance (Q)
        q_ca = np.array([
            [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
            [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 3 / 6, dt ** 2 / 2, dt]
        ])
        q_r = q_ca * process_noise[0] ** 2
        q_d = q_ca * process_noise[1] ** 2
        self.Q = np.block([
            [q_r, np.zeros((3, 3))],
            [np.zeros((3, 3)), q_d]
        ])

        # Measurement Noise Covariance (R)
        self.R = np.diag(np.array(meas_noise) ** 2)

    def predict(self):
        # Predict state
        self.x = self.F @ self.x
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Measurement residual
        y = z - self.H @ self.x
        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P




#===================================================================================================
#========================<<< The Track class storing info on a single track >>>=====================
#===================================================================================================
class Track:
    """
    Represents a single tracked target.
    Contains all state information requested by the user.

    Separates the KF's internal smoothed state (used for prediction)
    from the last associated measurement (used for data extraction).
    """

    def __init__(self, measurement, track_id, dt, process_noise, meas_noise):
        self.id = track_id  # Requirement 1: Unique track ID
        self.kf = KalmanFilter_CA_RD(dt, process_noise, meas_noise)

        # Initialize state with the first measurement
        self.kf.x[0] = measurement[0]  # r
        self.kf.x[3] = measurement[1]  # d

        # --- Track State Metrics ---
        self.age = 1  # Requirement 2: Number of frames track exists
        self.hits_total = 1  # Total number of associations
        self.consecutive_hits = 1  # Requirement 3: Consecutive associations
        self.misses_consecutive = 0  # Tracks consecutive misses (for deletion)
        self.misses_total = 0  # Requirement 4: Total missed frames

        self.is_confirmed = False

        # --- Store the last "ground truth" measurement ---
        self.last_measurement = measurement

    @property
    def state(self):
        """Returns the full 6x1 state vector [r, r_dot, r_ddot, d, d_dot, d_ddot]"""
        return self.kf.x.flatten()

    @property
    def position(self):
        """
        Returns the KF's internal 2D predicted position [r, d] (float).
        *** This is used by the Tracker for association. ***
        """
        return np.array([self.kf.x[0], self.kf.x[3]]).flatten()

    @property
    def int_position(self):
        """
        Returns the KF's internal 2D predicted position as rounded integers.
        *** This is used for debug printing. ***
        """
        return np.round(self.position).astype(int)

    @property
    def measurement_position(self):
        """
        Returns the last associated raw measurement [r, d] as rounded integers.
        *** This is used for data extraction. ***
        """
        return np.round(self.last_measurement).astype(int)

    def predict(self):
        """Called every frame. Predicts state and increments age."""
        self.kf.predict()
        self.age += 1

    def update(self, measurement):
        """Called when a measurement is associated with this track."""
        # 1. Update the KF's internal state for the *next* prediction
        self.kf.update(measurement.reshape(2, 1))

        # 2. Store the raw measurement as the "ground truth" position
        self.last_measurement = measurement

        # 3. Update track metrics
        self.hits_total += 1
        self.consecutive_hits += 1
        self.misses_consecutive = 0

    def mark_missed(self):
        """Called when no measurement is associated with this track."""
        self.consecutive_hits = 0
        self.misses_consecutive += 1
        self.misses_total += 1



#===================================================================================================
#=======================<<< The main tracker class, tracker API for the code >>>====================
#===================================================================================================
class Tracker:
    """
    Manages multiple tracks using a Kalman Filter and data association.
    """
    def __init__(self, dist_thresh, max_misses, min_hits_confirm, dt, process_noise, meas_noise):
        # --- Tracker Settings ---
        self.dist_thresh = dist_thresh
        self.max_misses = max_misses
        self.min_hits_confirm = min_hits_confirm

        # Parameters for new KFs
        self.dt = dt
        self.process_noise = process_noise
        self.meas_noise = meas_noise

        self.tracks = []
        self.next_track_id = 0

    def process_frame(self, measurements):
        """
        Main function to process detections from a single frame.
        measurements: (N_detections, 2) numpy array of [r, d]
        """

        # 1. Predict all existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Associate measurements to tracks
        if measurements.shape[0] > 0 and len(self.tracks) > 0:
            cost_matrix = np.zeros((len(self.tracks), measurements.shape[0]))
            for i, track in enumerate(self.tracks):
                cost_matrix[i, :] = np.linalg.norm(track.position - measurements, axis=1)

            cost_matrix[cost_matrix > self.dist_thresh] = 1e8
            track_indices, meas_indices = linear_sum_assignment(cost_matrix)

            valid_matches = cost_matrix[track_indices, meas_indices] <= self.dist_thresh
            track_indices_matched = track_indices[valid_matches]
            meas_indices_matched = meas_indices[valid_matches]

            unmatched_tracks = set(range(len(self.tracks))) - set(track_indices_matched)
            unmatched_measurements = set(range(measurements.shape[0])) - set(meas_indices_matched)

        else:
            unmatched_tracks = set(range(len(self.tracks)))
            unmatched_measurements = set(range(measurements.shape[0]))
            track_indices_matched = []
            meas_indices_matched = []

        # 3. Update matched tracks
        for track_idx, meas_idx in zip(track_indices_matched, meas_indices_matched):
            track = self.tracks[track_idx]
            track.update(measurements[meas_idx])

            # Confirmation based on consecutive hits
            if not track.is_confirmed and track.consecutive_hits >= self.min_hits_confirm:
                track.is_confirmed = True

        # 4. Manage unmatched tracks
        tracks_to_delete = []
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed()

            if track.misses_consecutive > self.max_misses:
                tracks_to_delete.append(track)

        self.tracks = [t for t in self.tracks if t not in tracks_to_delete]

        # 5. Create new tracks for unmatched measurements
        for meas_idx in unmatched_measurements:
            new_track = Track(measurements[meas_idx], self.next_track_id, self.dt, self.process_noise, self.meas_noise)
            self.tracks.append(new_track)
            self.next_track_id += 1

    def get_confirmed_tracks(self):
        """Returns a list of all currently active and confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed]