import numpy as np

from wavedispx.egf_reference import PassiveEgfReference
from wavedispx.processing import NCFBranches, NCFGather
from wavedispx.supervised_oracle import fit_supervised_egf_oracle_weights


def test_supervised_oracle_prefers_window_matching_known_reference():
    time = np.arange(21, dtype=np.float64)
    distance = np.asarray((20.0, 40.0))
    target = np.column_stack(
        (
            np.exp(-0.5 * ((time - 5.0) / 1.0) ** 2),
            np.exp(-0.5 * ((time - 9.0) / 1.0) ** 2),
        )
    )
    contamination = np.column_stack(
        (
            np.exp(-0.5 * ((time - 2.0) / 0.7) ** 2),
            np.exp(-0.5 * ((time - 3.0) / 0.7) ** 2),
        )
    )

    def branches(data):
        gather = NCFGather(
            data=data,
            lag_time_s=time,
            distance_m=distance,
            pair_count=np.ones(2, dtype=np.int64),
            valid_windows=1,
        )
        return NCFBranches(gather, gather, gather)

    reference = PassiveEgfReference(
        time_s=time,
        distance_m=distance,
        components=target[:, :, None],
        component_names=("zz",),
        pair_i=np.asarray((0, 0)),
        pair_j=np.asarray((1, 2)),
        pair_distance_index=np.asarray((0, 1)),
        source_time_function="unit_impulse",
        calibration_operator="identity",
        delay_removed_s=0.0,
        maximum_frequency_hz=0.5,
        build_seconds=0.0,
    )
    result = fit_supervised_egf_oracle_weights(
        [branches(target), branches(contamination)],
        reference,
        component_name="zz",
        frequency_range_hz=(0.01, 0.49),
        minimum_velocity_mps=2.0,
        maximum_velocity_mps=100.0,
        minimum_effective_fraction=0.5,
    )
    assert result.positive[0] > result.positive[1]
    assert result.negative[0] > result.negative[1]
    assert result.positive_effective_window_count >= 1.0 - 1e-10
    assert result.negative_effective_window_count >= 1.0 - 1e-10
