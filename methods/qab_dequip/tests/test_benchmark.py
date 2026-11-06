"""Check benchmark plumbing with tiny synthetic fixtures, not MATLAB evidence."""

import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from quantum_denoising.benchmark import array_digest, compare_matlab, run_benchmark
from quantum_denoising.cli import parser, main


def test_signal_benchmark_exports_shared_arrays(tmp_path):
    source = tmp_path / "signal.mat"
    output = tmp_path / "run"
    savemat(source, {"S": (3 + np.sin(np.arange(16.0)))[None, :]})
    args = parser().parse_args(["benchmark-signal", "--signal", str(source), "--out", str(output),
                               "--repeats", "1", "--warmups", "0", "--qab-slopes", "2"])
    run_benchmark(args)
    record = json.loads((output / "results.json").read_text(encoding="utf-8"))["results"][0]
    shared = loadmat(output / "case_signal_16.mat", simplify_cells=True)
    assert record["input_sha256"] == shared["input_sha256"]
    np.testing.assert_array_equal(shared["qab_signal_python"], np.load(output / "signal_16_qab_signal.npy"))
    assert len(record["timing"]["seconds"]) == 1
    assert record["timing"]["median_seconds"] > 0
    assert record["input_quality"]["snr_db"] == pytest.approx(15)
    assert record["oracle"] is True
    with pytest.raises(FileExistsError):
        run_benchmark(args)


def test_comparison_requires_real_baseline_file(tmp_path):
    with pytest.raises(ValueError, match="No matching MATLAB"):
        compare_matlab(str(tmp_path))


def test_comparison_schema_with_explicitly_synthetic_fixture(tmp_path):
    # This checks reporting logic ONLY. Values are artificial unit-test inputs.
    clean = np.arange(16.0)
    output = clean + 0.1
    shared = {"clean": clean, "noisy": output, "peak": 15,
              "input_sha256": array_digest(output), "qab_signal_config": {"slopes": 3}}
    savemat(tmp_path / "case_unit_test.mat", {**shared,
            "qab_signal_python": output,
            "qab_signal_python_seconds": [2.0, 3.0, 4.0]})
    savemat(tmp_path / "case_unit_test_matlab.mat", {**shared,
            "qab_signal_matlab": output + 1e-8, "qab_signal_seconds": [5.0, 6.0, 7.0],
            "matlab_version": "SYNTHETIC_UNIT_TEST_NOT_A_MATLAB_RUN"})
    result = compare_matlab(str(tmp_path))
    assert result[0]["within_tolerance"]
    assert result[0]["python_over_matlab"] == 0.5
    assert result[0]["max_abs_error"] == pytest.approx(1e-8)


@pytest.mark.parametrize("corruption", ["clean", "noisy", "config", "time", "tolerance"])
def test_comparison_rejects_mismatched_evidence(tmp_path, corruption):
    test_comparison_schema_with_explicitly_synthetic_fixture(tmp_path)
    path = tmp_path / "case_unit_test_matlab.mat"
    baseline = loadmat(path, simplify_cells=True)
    if corruption in ("clean", "noisy"):
        baseline[corruption][0] += 1
    elif corruption == "config":
        baseline["qab_signal_config"]["slopes"] = 20
    elif corruption == "time":
        baseline["qab_signal_seconds"] = [-1, 6, 7]
    savemat(path, {k: v for k, v in baseline.items() if not k.startswith("__")})
    with pytest.raises(ValueError):
        compare_matlab(str(tmp_path), float("nan") if corruption == "tolerance" else 1e-6)


def test_qab_cli_requires_threshold_and_does_not_overwrite(tmp_path):
    source = tmp_path / "input.npy"
    np.save(source, np.ones((8, 8)))
    with pytest.raises(SystemExit) as error:
        main(["denoise", "--image", str(source), "--algorithm", "qab", "--out", str(tmp_path / "out.npy")])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        main(["denoise", "--image", str(source), "--out", str(source)])
    np.testing.assert_array_equal(np.load(source), np.ones((8, 8)))

