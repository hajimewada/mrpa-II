from __future__ import annotations

import numpy as np
import soundfile as sf

from analysis.f0.extract_real_f0 import extract_f0, load_audio


def test_extracts_f0_from_sine_wave(tmp_path):
    sample_rate = 16_000
    duration = 1.0
    expected_f0 = 220.0

    time = np.arange(
        int(sample_rate * duration),
        dtype=np.float64,
    ) / sample_rate

    audio = 0.2 * np.sin(2.0 * np.pi * expected_f0 * time)

    input_path = tmp_path / "sine_220hz.wav"
    sf.write(input_path, audio, sample_rate)

    loaded_audio, loaded_sample_rate, channels = load_audio(input_path)

    f0, time_axis = extract_f0(
        loaded_audio,
        loaded_sample_rate,
        method="dio",
        frame_period=5.0,
        f0_floor=50.0,
        f0_ceil=800.0,
    )

    voiced_f0 = f0[f0 > 0.0]

    assert loaded_sample_rate == sample_rate
    assert channels == 1
    assert f0.ndim == 1
    assert f0.shape == time_axis.shape
    assert voiced_f0.size > 0
    assert abs(float(np.median(voiced_f0)) - expected_f0) < 5.0
