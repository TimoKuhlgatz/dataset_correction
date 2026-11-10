"""One module per error, named by its number in the agreed order of correction (eNN_<name>.py).

Each module has
    check_for_<error>(recording)        detects the error in one recording (or session): a CheckResult with
                                        detected True / False / None (undecided), a plain-language summary,
                                        the evidence per segment and the thresholds used;
    remove_<error>(recording, result)   a corrected copy with one more entry in its correction log;
    handle_<error>(datasets, data=...)  the check over whole datasets: an ErrorReport with the prevalence per
                                        dataset, the per-trial rows, notes and the corrected data (_report.py).

The check of error N runs on data corrected for every error with a smaller number (mocapref.correction):
     1 swapped marker labels      2 marker in the wrong place      3 reference jumps or flips
     4 gaps filled without a flag 5 clipping                       6 integer overflow
     7 dead, stuck or railed      8 wrong trial pairing            9 IMU on the wrong segment
    10 mis-assembled streams     11 sampling rate not as declared 12 time offset IMU vs reference
    13 clock drift               14 IMUs on different time scales 15 corrupted single samples
    16 mirrored gyroscope        17 gyroscope offset              18 mounting rotation
    19 reference vertical        20 accelerometer negated         21 accelerometer bias
    22 gravity removed
"""
from ._result import CheckResult
from .e19_e20_accelerometer_sign import check_for_negated_accelerometer, remove_negated_accelerometer
from .e22_gravity_removed import check_for_gravity_removed, restore_removed_gravity
from .e05_clipping import check_for_cut_off_peaks, remove_cut_off_peaks
from .e11_sampling_rate import check_for_effective_rate, check_for_time_scale, remove_time_scale
from .e13_clock_drift import check_for_clock_drift, remove_clock_drift
from .e06_integer_overflow import check_for_integer_overflow, remove_integer_overflow
from .e17_gyroscope_offset import check_for_gyroscope_offset, remove_gyroscope_offset
from .e18_mounting import check_for_mounting_in_convergence, check_for_mounting_misalignment, remove_mounting_misalignment
from .e04_gap_fill import check_for_gap_fill, remove_gap_fill
from .e01_label_swaps import check_for_label_swaps, remove_label_swaps
from .e02_misplaced_markers import check_for_misplaced_markers, remove_misplaced_markers
from .e03_reference_jumps import check_for_reference_jumps, remove_reference_jumps
from .e14_time_scales import check_for_sensor_rate_mismatch, remove_sensor_rate_mismatch
from .e09_segment_assignment import check_for_segment_assignment, remove_segment_assignment
from .e15_corrupted_samples import check_for_corrupted_samples, remove_corrupted_samples
from .e07_dead_channels import (check_for_dead_channels, check_for_held_samples, remove_dead_channels,
                                remove_held_samples)
from .e16_mirrored_gyroscope import check_for_mirrored_gyroscope, remove_mirrored_gyroscope
from .e12_time_offset import check_for_synchronisation_offset, remove_synchronisation_offset

__all__ = [
    'CheckResult',
    'check_for_negated_accelerometer', 'remove_negated_accelerometer',
    'check_for_gravity_removed', 'restore_removed_gravity',
    'check_for_segment_assignment', 'remove_segment_assignment',
    'check_for_synchronisation_offset', 'remove_synchronisation_offset',
    'check_for_clock_drift', 'remove_clock_drift',
    'check_for_time_scale', 'remove_time_scale', 'check_for_effective_rate',
    'check_for_sensor_rate_mismatch', 'remove_sensor_rate_mismatch',
    'check_for_cut_off_peaks', 'remove_cut_off_peaks',
    'check_for_integer_overflow', 'remove_integer_overflow',
    'check_for_held_samples', 'remove_held_samples', 'check_for_dead_channels', 'remove_dead_channels',
    'check_for_corrupted_samples', 'remove_corrupted_samples',
    'check_for_gap_fill', 'remove_gap_fill', 'check_for_label_swaps', 'remove_label_swaps',
    'check_for_misplaced_markers', 'remove_misplaced_markers',
    'check_for_reference_jumps', 'remove_reference_jumps',
    'check_for_gyroscope_offset', 'remove_gyroscope_offset',
    'check_for_mirrored_gyroscope',
    'remove_mirrored_gyroscope',
    'check_for_mounting_misalignment', 'remove_mounting_misalignment',
]

# handlers 1-10
from .e01_label_swaps import handle_label_swaps  # noqa: E402
from .e02_misplaced_markers import handle_misplaced_markers  # noqa: E402
from .e03_reference_jumps import handle_reference_jumps  # noqa: E402
from .e04_gap_fill import handle_gap_fill  # noqa: E402
from .e05_clipping import handle_clipping  # noqa: E402
from .e06_integer_overflow import handle_integer_overflow  # noqa: E402
from .e07_dead_channels import handle_dead_channels  # noqa: E402
from .e08_trial_pairing import handle_trial_pairing  # noqa: E402
from .e09_segment_assignment import handle_segment_assignment  # noqa: E402
from .e10_mis_assembled_streams import handle_mis_assembled_streams  # noqa: E402

__all__ += [
    'handle_label_swaps', 'handle_misplaced_markers', 'handle_reference_jumps', 'handle_gap_fill', 'handle_clipping',
    'handle_integer_overflow', 'handle_dead_channels', 'handle_trial_pairing', 'handle_segment_assignment',
    'handle_mis_assembled_streams',
]

# handlers 11-22
from .e11_sampling_rate import handle_sampling_rate  # noqa: E402
from .e12_time_offset import handle_time_offset  # noqa: E402
from .e13_clock_drift import handle_clock_drift  # noqa: E402
from .e14_time_scales import handle_time_scales  # noqa: E402
from .e15_corrupted_samples import handle_corrupted_samples  # noqa: E402
from .e16_mirrored_gyroscope import handle_mirrored_gyroscope  # noqa: E402
from .e17_gyroscope_offset import handle_gyroscope_offset  # noqa: E402
from .e18_mounting import handle_mounting  # noqa: E402
from .e19_e20_accelerometer_sign import handle_negated_accelerometer, handle_reference_vertical  # noqa: E402
from .e21_accelerometer_bias import handle_accelerometer_bias  # noqa: E402
from .e22_gravity_removed import handle_gravity_removed  # noqa: E402

__all__ += [
    'handle_sampling_rate', 'handle_time_offset', 'handle_clock_drift', 'handle_time_scales',
    'handle_corrupted_samples', 'handle_mirrored_gyroscope', 'handle_gyroscope_offset', 'handle_mounting',
    'handle_reference_vertical', 'handle_negated_accelerometer', 'handle_accelerometer_bias',
    'handle_gravity_removed',
]
