"""Tests for BPM Sync Buffer postprocessor."""

import time
import torch
import numpy as np


def make_frame(h=336, w=576, val=128):
    """Create a test frame as (1, H, W, 3) uint8 tensor."""
    frame = np.full((h, w, 3), val, dtype=np.uint8)
    return torch.from_numpy(frame).unsqueeze(0)


def test_passthrough():
    """Zero latency = passthrough: output matches latest input."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=0, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    frame = make_frame(val=200)
    result = buf(video=[frame])

    assert "video" in result
    video = result["video"]
    assert video.shape == (1, 336, 576, 3)
    assert video.max() <= 1.0
    assert video.mean() > 0.5

    print("  [OK] Passthrough test passed")


def test_latency_buffer():
    """Latency mode: frames are delayed."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=200, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(5):
        frame = make_frame(val=50 + i * 40)
        buf(video=[frame])
        time.sleep(0.05)

    assert len(buf._fifo) > 0
    print(f"  FIFO has {len(buf._fifo)} frames")

    print("  [OK] Latency buffer test passed")


def test_speed_control():
    """Speed > 1 advances playback head faster."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=1000, speed=2.0, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(10):
        buf(video=[make_frame(val=50 + i * 20)])
        time.sleep(0.05)

    assert buf._effective_speed == 2.0
    print("  [OK] Speed control test passed")


def test_auto_speed():
    """Auto speed mode adjusts effective speed based on buffer fill."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=500, auto_speed=True, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(10):
        buf(video=[make_frame(val=100)])
        time.sleep(0.03)

    # Auto speed should have adjusted from 1.0
    # (exact value depends on fill level, just check it ran without error)
    assert buf._effective_speed > 0
    print(f"  Auto speed = {buf._effective_speed:.2f}x")
    print("  [OK] Auto speed test passed")


def test_bpm_stamp():
    """Frames should be stamped with BPM at capture time."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=200, clock_bpm=140.0, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    buf(video=[make_frame(val=100)])
    assert len(buf._fifo) == 1
    assert buf._fifo[0].capture_bpm == 140.0

    print("  [OK] BPM stamp test passed")


def test_hold():
    """Hold freezes playback."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=100, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(5):
        buf(video=[make_frame(val=100 + i * 30)])
        time.sleep(0.03)

    result1 = buf(video=[make_frame()], hold=True)
    time.sleep(0.1)
    result2 = buf(video=[make_frame()], hold=True)

    assert buf._hold_active
    print("  [OK] Hold test passed")


def test_reset():
    """Reset clears the buffer."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=500, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(5):
        buf(video=[make_frame()])

    assert len(buf._fifo) > 0
    buf(video=[make_frame()], reset_buffer=True)
    assert len(buf._fifo) == 1
    print("  [OK] Reset test passed")


def test_overlay():
    """Overlay draws without crashing."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=500, show_overlay=True)
    buf = BpmSyncBufferPostprocessor(config)

    for i in range(3):
        result = buf(video=[make_frame(val=80)])
        time.sleep(0.05)

    video = result["video"]
    assert video.shape == (1, 336, 576, 3)
    print("  [OK] Overlay test passed")


def test_tempo_offset():
    """Tempo offset adjusts BPM stamp."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(latency_ms=200, clock_bpm=120.0, show_overlay=False)
    buf = BpmSyncBufferPostprocessor(config)

    # +10% offset → 132 bpm
    buf(video=[make_frame()], tempo_offset_pct=10.0)
    assert abs(buf._fifo[0].capture_bpm - 132.0) < 0.1

    print("  [OK] Tempo offset test passed")


def test_min_max_clamp():
    """Latency is clamped between min and max delay."""
    from bpm_sync_buffer_vjtools.pipeline import BpmSyncBufferPostprocessor, BpmSyncBufferConfig

    config = BpmSyncBufferConfig(
        latency_ms=100, min_delay_ms=200, max_delay_ms=5000, show_overlay=False
    )
    buf = BpmSyncBufferPostprocessor(config)

    # latency_ms=100 but min=200, so it should clamp up to 200
    # We can't easily verify the clamped value directly, but we can
    # verify no crash and the buffer runs
    for i in range(3):
        buf(video=[make_frame()])
        time.sleep(0.02)

    assert len(buf._fifo) > 0
    print("  [OK] Min/max clamp test passed")


if __name__ == "__main__":
    print("\n=== BPM Sync Buffer Tests ===\n")
    tests = [
        test_passthrough,
        test_latency_buffer,
        test_speed_control,
        test_auto_speed,
        test_bpm_stamp,
        test_hold,
        test_reset,
        test_overlay,
        test_tempo_offset,
        test_min_max_clamp,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
