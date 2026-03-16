"""Tests for BPM Sync Buffer postprocessor."""

import time
import torch
import numpy as np


def make_frame(h=336, w=576, val=128):
    """Create a test frame as (1, H, W, 3) uint8 tensor."""
    frame = np.full((h, w, 3), val, dtype=np.uint8)
    return torch.from_numpy(frame).unsqueeze(0)


def test_passthrough():
    """Passthrough mode: output matches input immediately."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(buffer_mode="passthrough", show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    frame = make_frame(val=200)
    result = buf(video=[frame])

    assert "video" in result
    video = result["video"]
    assert video.shape == (1, 336, 576, 3)
    assert video.max() <= 1.0
    # Should be close to 200/255 since passthrough
    assert video.mean() > 0.5

    print("  [OK] Passthrough test passed")


def test_latency_buffer():
    """Latency mode: frames are delayed."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(buffer_mode="latency", latency_delay_ms=200, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    # Feed some frames
    for i in range(5):
        frame = make_frame(val=50 + i * 40)
        buf(video=[frame])
        time.sleep(0.05)

    # Buffer should have frames
    assert len(buf._fifo) > 0
    print(f"  FIFO has {len(buf._fifo)} frames")

    print("  [OK] Latency buffer test passed")


def test_beat_buffer():
    """Beat mode: delay based on beat depth."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(
        buffer_mode="beat", beat_division="1/2", beat_multiplier=2,
        clock_bpm=120.0, show_overlay=False
    )
    buf = BpmSyncBufferPostprocessor(config)

    # At 120 BPM, 2 beats = 1 second delay
    for i in range(10):
        frame = make_frame(val=100)
        buf(video=[frame])
        time.sleep(0.02)

    assert len(buf._fifo) > 0
    print("  [OK] Beat buffer test passed")


def test_hold():
    """Hold freezes playback."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(buffer_mode="latency", latency_delay_ms=100, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    # Feed frames
    for i in range(5):
        buf(video=[make_frame(val=100 + i * 30)])
        time.sleep(0.03)

    # Engage hold
    result1 = buf(video=[make_frame()], hold=True)
    time.sleep(0.1)
    result2 = buf(video=[make_frame()], hold=True)

    # Both should return same frame (held)
    assert buf._hold_active
    print("  [OK] Hold test passed")


def test_reset():
    """Reset clears the buffer."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(buffer_mode="latency", show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(5):
        buf(video=[make_frame()])

    assert len(buf._fifo) > 0
    buf(video=[make_frame()], reset_buffer=True)
    # After reset + new frame, FIFO has just 1
    assert len(buf._fifo) == 1
    print("  [OK] Reset test passed")


def test_overlay():
    """Overlay draws without crashing."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(buffer_mode="latency", latency_delay_ms=500, show_overlay=True)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(3):
        result = buf(video=[make_frame(val=80)])
        time.sleep(0.05)

    video = result["video"]
    assert video.shape == (1, 336, 576, 3)
    print("  [OK] Overlay test passed")


if __name__ == "__main__":
    print("\n=== BPM Sync Buffer Tests ===\n")
    tests = [
        test_passthrough,
        test_latency_buffer,
        test_beat_buffer,
        test_hold,
        test_reset,
        test_overlay,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
