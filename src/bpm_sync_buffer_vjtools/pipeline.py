"""
BPM Sync Buffer — Adjustable Latency Buffer for Daydream Scope

Postprocessor that buffers AI-generated frames for smooth, consistent playback.
Instead of bursty chunk output (12 frames at once, then nothing), the buffer
accumulates frames and releases them at a steady rate.

Controls (all MIDI-mappable):
  - Latency:       Delay fader from min_delay to max_delay (default 0–60 000 ms)
  - Speed:         Playback speed multiplier (0.25× – 4.0×).  Auto mode adjusts
                   speed to keep buffer fill at a target level.
  - BPM:           When known, frames are BPM-stamped at ingest.  If playback BPM
                   differs from capture BPM, speed auto-compensates.
  - Tempo Offset:  Manual ±% nudge if detected BPM is wrong.
  - Hold:          Freeze output on current frame.
  - Reset:         Flush buffer.

Visual overlay shows buffer fill level, mode, speed, FPS, and delay.

Clock sources (for BPM detection):
  - Ableton Link:  Networked beat sync with DAWs and other Link-enabled apps
  - MIDI Clock:    Standard MIDI timing (24 PPQN) from DJ software, drum machines
  - OSC:           Beat position and BPM pushed from external software
  - Internal:      Free-running clock at configured BPM (fallback)
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Clock Sources ──────────────────────────────────────────────────────────

class ClockSource(str, Enum):
    LINK = "link"
    MIDI_CLOCK = "midi_clock"
    OSC = "osc"
    INTERNAL = "internal"


class LinkClock:
    """Thin wrapper around aalink — runs async event loop in background thread."""

    def __init__(self, initial_bpm: float = 120.0):
        self._beat = 0.0
        self._tempo = initial_bpm
        self._phase = 0.0
        self._num_peers = 0
        self._enabled = False
        self._link = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def beat(self) -> float:
        return self._beat

    @property
    def tempo(self) -> float:
        return self._tempo

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def num_peers(self) -> int:
        return self._num_peers

    def start(self, bpm: float = 120.0):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, args=(bpm,), daemon=True, name="link-clock"
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._enabled = False

    def _run_loop(self, bpm: float):
        try:
            from aalink import Link
        except ImportError:
            logger.warning("[Buffer/Link] aalink not installed — using free-running clock")
            self._run_freerunning(bpm)
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def poll():
            link = Link(bpm)
            link.enabled = True
            self._link = link
            self._enabled = True
            self._tempo = bpm
            while not self._stop_event.is_set():
                try:
                    beat_val = await asyncio.wait_for(link.sync(1 / 16), timeout=0.1)
                    self._beat = beat_val
                    self._tempo = bpm
                    self._phase = beat_val % 4.0
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    await asyncio.sleep(0.016)
            link.enabled = False
            self._enabled = False

        try:
            loop.run_until_complete(poll())
        finally:
            loop.close()

    def _run_freerunning(self, bpm: float):
        self._enabled = True
        self._tempo = bpm
        start = time.monotonic()
        while not self._stop_event.is_set():
            self._beat = (time.monotonic() - start) * (bpm / 60.0)
            self._phase = self._beat % 4.0
            time.sleep(0.008)
        self._enabled = False


class ClockManager:
    """Unified clock abstraction over Link / MIDI / OSC / internal sources."""

    def __init__(self, initial_bpm: float = 120.0):
        self._source = ClockSource.INTERNAL
        self._link_clock: Optional[LinkClock] = None
        self._midi_clock = None
        self._internal_bpm = initial_bpm
        self._internal_start = time.monotonic()
        self._osc_beat: float = 0.0
        self._osc_bpm: float = initial_bpm
        self._midi_device = ""

    @property
    def source(self) -> ClockSource:
        return self._source

    @property
    def beat(self) -> float:
        if self._source == ClockSource.LINK and self._link_clock:
            return self._link_clock.beat
        elif self._source == ClockSource.MIDI_CLOCK and self._midi_clock:
            return self._midi_clock.beat
        elif self._source == ClockSource.OSC:
            return self._osc_beat
        else:
            elapsed = time.monotonic() - self._internal_start
            return elapsed * (self._internal_bpm / 60.0)

    @property
    def tempo(self) -> float:
        if self._source == ClockSource.LINK and self._link_clock:
            return self._link_clock.tempo
        elif self._source == ClockSource.MIDI_CLOCK and self._midi_clock:
            t = self._midi_clock.tempo
            return t if t > 0 else self._internal_bpm
        elif self._source == ClockSource.OSC:
            return self._osc_bpm
        else:
            return self._internal_bpm

    @property
    def source_info(self) -> dict:
        return {"source": self._source.value, "beat": self.beat, "tempo": self.tempo}

    def set_source(self, source: ClockSource, bpm: float = 120.0, midi_device: str = ""):
        if source == self._source and midi_device == self._midi_device:
            if source == ClockSource.INTERNAL:
                self._internal_bpm = bpm
            return
        self._stop_current()
        self._source = source
        self._midi_device = midi_device
        if source == ClockSource.LINK:
            self._link_clock = LinkClock(bpm)
            self._link_clock.start(bpm)
        elif source == ClockSource.MIDI_CLOCK:
            try:
                from .midi_clock import MidiClock
                self._midi_clock = MidiClock()
                self._midi_clock.start(device_name=midi_device)
            except ImportError:
                logger.warning("[Buffer] mido not installed — falling back to internal")
                self._source = ClockSource.INTERNAL
                self._internal_bpm = bpm
        elif source == ClockSource.INTERNAL:
            self._internal_bpm = bpm
            self._internal_start = time.monotonic()
        logger.info(f"[Buffer] Clock → {source.value}")

    def set_internal_bpm(self, bpm: float):
        self._internal_bpm = max(20.0, min(999.0, bpm))

    def update_osc(self, beat: float, bpm: float):
        self._osc_beat = beat
        if bpm > 0:
            self._osc_bpm = bpm

    def stop(self):
        self._stop_current()

    def _stop_current(self):
        if self._link_clock:
            self._link_clock.stop()
            self._link_clock = None
        if self._midi_clock:
            self._midi_clock.stop()
            self._midi_clock = None


# ─── Scope SDK imports ─────────────────────────────────────────────────────

try:
    from scope.core.pipelines.interface import Pipeline, Requirements
    from scope.core.pipelines.base_schema import (
        BasePipelineConfig, UsageType, ModeDefaults, ui_field_config,
    )
    _HAS_SCOPE = True
except ImportError:
    class Pipeline:
        pass
    class Requirements:
        def __init__(self, input_size: int = 1):
            self.input_size = input_size
    class BasePipelineConfig:
        pass
    class UsageType:
        POSTPROCESSOR = "postprocessor"
    class ModeDefaults:
        def __init__(self, default=False):
            self.default = default
    def ui_field_config(**kwargs):
        return kwargs
    _HAS_SCOPE = False

try:
    from pydantic import Field
except ImportError:
    def Field(default=None, **kwargs):
        return default


# ─── Buffered Frame ────────────────────────────────────────────────────────

@dataclass
class _BufferedFrame:
    """A frame with wall-clock timestamp and BPM at capture time."""
    frame: np.ndarray        # (H, W, C) uint8
    timestamp: float         # time.monotonic() when received
    capture_bpm: float       # BPM at the time this frame was generated


# ─── Config ────────────────────────────────────────────────────────────────

if _HAS_SCOPE:
    class BpmSyncBufferConfig(BasePipelineConfig):
        """Latency buffer with playback speed control and BPM compensation."""
        pipeline_id: ClassVar[str] = "bpm_sync_buffer_vjtools"
        pipeline_name: ClassVar[str] = "BPM Sync Buffer (VJ.Tools)"
        pipeline_description: ClassVar[str] = (
            "Adjustable latency buffer with playback speed control. "
            "BPM-stamped frames auto-compensate when tempo changes. "
            "All faders MIDI-mappable."
        )
        supports_prompts: ClassVar[bool] = False
        modified: ClassVar[bool] = True
        usage: ClassVar[list] = [UsageType.POSTPROCESSOR]
        modes: ClassVar[dict] = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(default=True),
        }

        # ── Latency ───────────────────────────────────────────────────

        latency_ms: int = Field(
            default=0,
            ge=0,
            le=60000,
            json_schema_extra=ui_field_config(
                order=0,
                label="Latency (ms)",
                category="configuration",
            ),
        )

        min_delay_ms: int = Field(
            default=0,
            ge=0,
            le=60000,
            json_schema_extra=ui_field_config(
                order=1,
                label="Min Delay (ms)",
            ),
        )

        max_delay_ms: int = Field(
            default=60000,
            ge=0,
            le=60000,
            json_schema_extra=ui_field_config(
                order=2,
                label="Max Delay (ms)",
            ),
        )

        # ── Playback Speed ────────────────────────────────────────────

        speed: float = Field(
            default=1.0,
            ge=0.25,
            le=4.0,
            json_schema_extra=ui_field_config(
                order=3,
                label="Speed",
                category="configuration",
            ),
        )

        auto_speed: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=4,
                label="Auto Speed",
                category="configuration",
            ),
        )

        auto_speed_target: float = Field(
            default=0.5,
            ge=0.1,
            le=0.9,
            json_schema_extra=ui_field_config(
                order=5,
                label="Auto Target Fill",
                category="configuration",
            ),
        )

        # ── BPM / Tempo ──────────────────────────────────────────────

        tempo_offset_pct: float = Field(
            default=0.0,
            ge=-50.0,
            le=50.0,
            json_schema_extra=ui_field_config(
                order=6,
                label="Tempo Offset %",
                category="configuration",
            ),
        )

        # ── Transport ────────────────────────────────────────────────

        hold: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=7,
                label="HOLD",
                category="configuration",
            ),
        )

        reset_buffer: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=8,
                label="Reset",
                category="configuration",
            ),
        )

        show_overlay: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=9,
                label="Show Overlay",
                category="configuration",
            ),
        )

        # ── Clock config ─────────────────────────────────────────────

        clock_source: ClockSource = Field(
            default=ClockSource.INTERNAL,
            json_schema_extra=ui_field_config(
                order=10,
                label="Clock Source",
            ),
        )

        clock_bpm: float = Field(
            default=120.0,
            ge=20.0,
            le=999.0,
            json_schema_extra=ui_field_config(
                order=11,
                label="BPM",
                category="configuration",
            ),
        )

        midi_device: str = Field(
            default="",
            json_schema_extra=ui_field_config(
                order=12,
                label="MIDI Clock Device",
            ),
        )

        osc_beat: float = Field(
            default=0.0,
            ge=0.0,
            json_schema_extra=ui_field_config(
                order=13,
                label="OSC Beat",
                category="configuration",
            ),
        )
else:
    class BpmSyncBufferConfig:
        """Standalone config for testing outside Scope."""
        def __init__(self, **kwargs):
            self.pipeline_id = kwargs.get("pipeline_id", "bpm_sync_buffer_vjtools")
            self.latency_ms = kwargs.get("latency_ms", 0)
            self.min_delay_ms = kwargs.get("min_delay_ms", 0)
            self.max_delay_ms = kwargs.get("max_delay_ms", 60000)
            self.speed = kwargs.get("speed", 1.0)
            self.auto_speed = kwargs.get("auto_speed", False)
            self.auto_speed_target = kwargs.get("auto_speed_target", 0.5)
            self.tempo_offset_pct = kwargs.get("tempo_offset_pct", 0.0)
            self.hold = kwargs.get("hold", False)
            self.reset_buffer = kwargs.get("reset_buffer", False)
            self.show_overlay = kwargs.get("show_overlay", False)
            self.clock_source = kwargs.get("clock_source", "internal")
            self.clock_bpm = kwargs.get("clock_bpm", 120.0)
            self.midi_device = kwargs.get("midi_device", "")
            self.osc_beat = kwargs.get("osc_beat", 0.0)


# ─── Postprocessor Pipeline ───────────────────────────────────────────────

class BpmSyncBufferPostprocessor(Pipeline):
    """
    Adjustable latency buffer with playback speed control.

    Architecture:
      - FIFO of BPM-stamped frames with wall-clock timestamps
      - A virtual playback head that advances through the FIFO
      - Speed control: 1.0× = real-time, >1 = catching up, <1 = slowing down
      - Auto-speed: PD controller keeps buffer fill at target level
      - BPM compensation: if capture BPM ≠ playback BPM, speed auto-adjusts
        so musical timing is preserved

    The latency fader sets how far back in the buffer the playback head starts.
    At 0ms it's passthrough (latest frame). Crank it up and you're looking
    at frames from the past, with speed controlling how fast you move through them.

    Visual overlay shows:
      - Fill bar: red → yellow → green → cyan as buffer fills
      - Current delay, speed, BPM
      - Input FPS and buffer depth
      - HOLD indicator when frozen
    """

    MAX_FIFO_FRAMES = 1800  # ~60s at 30fps
    FALLBACK_BPM = 120.0

    # Auto-speed PD controller gains
    AUTO_KP = 1.5   # Proportional: how aggressively to correct
    AUTO_KD = 0.3   # Derivative: damping to prevent oscillation

    @classmethod
    def get_config_class(cls):
        return BpmSyncBufferConfig

    def __init__(self, config=None, device=None, dtype=torch.float16, **kwargs):
        if config is None:
            config = BpmSyncBufferConfig()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if (device or self.device).type == "cuda" else torch.float32

        # Clock
        initial_bpm = getattr(config, "clock_bpm", 120.0)
        self._clock = ClockManager(initial_bpm)
        source_str = getattr(config, "clock_source", "internal")
        try:
            source = ClockSource(str(source_str))
        except ValueError:
            source = ClockSource.INTERNAL
        self._clock.set_source(source, bpm=initial_bpm,
                               midi_device=getattr(config, "midi_device", ""))

        # FIFO
        self._fifo: list[_BufferedFrame] = []

        # Playback head — virtual time position in the FIFO
        self._playback_time: float = 0.0  # wall-clock time of current playback position
        self._last_call_time: float = 0.0  # when __call__ was last invoked
        self._current_output: Optional[np.ndarray] = None

        # Hold
        self._hold_active: bool = False

        # Auto-speed state
        self._prev_fill_error: float = 0.0
        self._effective_speed: float = 1.0

        # FPS tracking
        self._input_count = 0
        self._output_count = 0
        self._fps_timestamps: list[float] = []
        self._input_fps = 0.0

        logger.info(f"[Buffer] Initialized (clock={source.value})")

    def __del__(self):
        if hasattr(self, "_clock"):
            self._clock.stop()

    def prepare(self, **kwargs):
        """Accept one frame at a time from the main pipeline output."""
        if _HAS_SCOPE:
            return Requirements(input_size=1)
        return None

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video", [])

        if isinstance(video, list) and len(video) == 0:
            return {"video": torch.zeros(1, 1, 1, 3)}

        now = time.monotonic()

        # ── Read runtime params ───────────────────────────────────────
        latency_ms = int(kwargs.get("latency_ms", getattr(self.config, "latency_ms", 0)))
        min_delay = int(kwargs.get("min_delay_ms", getattr(self.config, "min_delay_ms", 0)))
        max_delay = int(kwargs.get("max_delay_ms", getattr(self.config, "max_delay_ms", 60000)))
        speed = float(kwargs.get("speed", getattr(self.config, "speed", 1.0)))
        auto_speed = bool(kwargs.get("auto_speed", getattr(self.config, "auto_speed", False)))
        auto_target = float(kwargs.get("auto_speed_target", getattr(self.config, "auto_speed_target", 0.5)))
        tempo_offset = float(kwargs.get("tempo_offset_pct", getattr(self.config, "tempo_offset_pct", 0.0)))
        show_overlay = bool(kwargs.get("show_overlay", getattr(self.config, "show_overlay", True)))
        reset = bool(kwargs.get("reset_buffer", getattr(self.config, "reset_buffer", False)))
        hold = bool(kwargs.get("hold", getattr(self.config, "hold", False)))

        # Clamp latency to [min, max]
        latency_ms = max(min_delay, min(max_delay, latency_ms))
        delay_s = latency_ms / 1000.0

        # ── Clock updates ─────────────────────────────────────────────
        clock_src_str = kwargs.get("clock_source", getattr(self.config, "clock_source", "internal"))
        clock_bpm = float(kwargs.get("clock_bpm", getattr(self.config, "clock_bpm", 120.0)))
        midi_dev = kwargs.get("midi_device", getattr(self.config, "midi_device", ""))

        try:
            target_src = ClockSource(str(clock_src_str))
        except ValueError:
            target_src = ClockSource.INTERNAL

        if target_src != self._clock.source:
            self._clock.set_source(target_src, bpm=clock_bpm, midi_device=midi_dev)
        elif target_src == ClockSource.INTERNAL:
            self._clock.set_internal_bpm(clock_bpm)

        if target_src == ClockSource.OSC:
            osc_beat = float(kwargs.get("osc_beat", getattr(self.config, "osc_beat", 0.0)))
            self._clock.update_osc(beat=osc_beat, bpm=clock_bpm)

        current_bpm = self._clock.tempo or self.FALLBACK_BPM
        # Apply manual tempo offset
        adjusted_bpm = current_bpm * (1.0 + tempo_offset / 100.0)

        # ── Reset ─────────────────────────────────────────────────────
        if reset:
            self._fifo.clear()
            self._current_output = None
            self._hold_active = False
            self._playback_time = 0.0
            self._last_call_time = 0.0
            self._prev_fill_error = 0.0
            self._effective_speed = 1.0
            logger.info("[Buffer] Reset")

        # ── Hold ──────────────────────────────────────────────────────
        if hold and not self._hold_active:
            self._hold_active = True
        elif not hold and self._hold_active:
            self._hold_active = False

        # ── Ingest frames ─────────────────────────────────────────────
        if isinstance(video, list):
            frames = torch.cat(video, dim=0).float()
        else:
            frames = video.float() if video.dim() == 4 else video.unsqueeze(0).float()

        if frames.max() <= 1.0:
            frames = frames * 255.0

        F, H, W, C = frames.shape

        # Track input FPS
        self._input_count += F
        self._fps_timestamps.append(now)
        self._fps_timestamps = [t for t in self._fps_timestamps if now - t < 3.0]
        if len(self._fps_timestamps) > 1:
            span = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if span > 0:
                self._input_fps = (len(self._fps_timestamps) - 1) / span

        # Add to FIFO with BPM stamp
        for f_idx in range(F):
            frame_np = frames[f_idx].cpu().numpy().astype(np.uint8)
            self._fifo.append(_BufferedFrame(
                frame=frame_np,
                timestamp=now,
                capture_bpm=adjusted_bpm,
            ))

        while len(self._fifo) > self.MAX_FIFO_FRAMES:
            self._fifo.pop(0)

        # ── Compute effective speed ───────────────────────────────────
        dt = now - self._last_call_time if self._last_call_time > 0 else 0.0
        self._last_call_time = now

        if delay_s <= 0 or not self._fifo:
            # Passthrough: no delay, just return latest frame
            output = self._fifo[-1].frame if self._fifo else np.zeros((H, W, C), dtype=np.uint8)
            self._effective_speed = 1.0
        elif self._hold_active:
            # Frozen: don't advance playback head
            output = self._pick_at_time(self._playback_time)
            if output is None:
                output = np.zeros((H, W, C), dtype=np.uint8)
        else:
            # ── Compute playback speed ────────────────────────────────
            # 1. Start with manual speed
            effective = speed

            # 2. BPM compensation: if the frame was captured at a different BPM,
            #    adjust speed so musical timing is preserved
            #    (playback_bpm / capture_bpm) ratio
            target_frame = self._find_frame_at_delay(delay_s, now)
            if target_frame is not None and target_frame.capture_bpm > 0:
                bpm_ratio = adjusted_bpm / target_frame.capture_bpm
                effective *= bpm_ratio

            # 3. Auto-speed: PD controller to maintain target fill level
            if auto_speed:
                fill = self._buffer_fill(delay_s)
                fill_error = fill - auto_target  # positive = too full, negative = depleting
                d_error = (fill_error - self._prev_fill_error) / max(dt, 0.001) if dt > 0 else 0.0
                self._prev_fill_error = fill_error

                auto_correction = 1.0 + (self.AUTO_KP * fill_error) + (self.AUTO_KD * d_error)
                auto_correction = max(0.25, min(4.0, auto_correction))
                effective *= auto_correction

            # Clamp final speed
            effective = max(0.1, min(8.0, effective))
            self._effective_speed = effective

            # ── Advance playback head ─────────────────────────────────
            # Initialize playback time if needed
            if self._playback_time <= 0:
                self._playback_time = now - delay_s

            # Advance by dt * effective_speed
            self._playback_time += dt * effective

            # Don't let playback head go past the newest frame
            if self._fifo:
                newest_t = self._fifo[-1].timestamp
                if self._playback_time > newest_t:
                    self._playback_time = newest_t

            # Don't let playback head go before the oldest frame
            if self._fifo:
                oldest_t = self._fifo[0].timestamp
                if self._playback_time < oldest_t:
                    self._playback_time = oldest_t

            output = self._pick_at_time(self._playback_time)
            if output is None:
                output = np.zeros((H, W, C), dtype=np.uint8)

        self._output_count += 1

        # ── Overlay ───────────────────────────────────────────────────
        if show_overlay:
            output = self._draw_overlay(
                output, delay_s, latency_ms, self._effective_speed,
                auto_speed, adjusted_bpm, tempo_offset,
            )

        out_tensor = torch.from_numpy(output).float().unsqueeze(0) / 255.0
        return {"video": out_tensor}

    # ── Helpers ─────────────────────────────────────────────────────────

    def _buffer_fill(self, delay_s: float) -> float:
        """Buffer fill ratio: 0.0 = empty, 1.0 = full relative to delay."""
        if not self._fifo or delay_s <= 0:
            return 0.0
        oldest = self._fifo[0].timestamp
        newest = self._fifo[-1].timestamp
        buffered_s = newest - oldest
        return min(1.0, buffered_s / delay_s)

    def _find_frame_at_delay(self, delay_s: float, now: float) -> Optional[_BufferedFrame]:
        """Find the frame closest to (now - delay_s) for BPM lookup."""
        if not self._fifo:
            return None
        target = now - delay_s
        # Quick bounds check
        if target <= self._fifo[0].timestamp:
            return self._fifo[0]
        if target >= self._fifo[-1].timestamp:
            return self._fifo[-1]
        # Binary search
        lo, hi = 0, len(self._fifo) - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if self._fifo[mid].timestamp < target:
                lo = mid + 1
            else:
                hi = mid
        return self._fifo[lo]

    def _pick_at_time(self, target_time: float) -> Optional[np.ndarray]:
        """Binary-search FIFO for the frame closest to target_time."""
        if not self._fifo:
            return self._current_output

        # Evict frames older than 2s before our target (keep some headroom)
        cutoff = target_time - 2.0
        while len(self._fifo) > 1 and self._fifo[0].timestamp < cutoff:
            self._fifo.pop(0)

        if not self._fifo:
            return self._current_output

        # Binary search
        lo, hi = 0, len(self._fifo) - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if self._fifo[mid].timestamp < target_time:
                lo = mid + 1
            else:
                hi = mid
        best = self._fifo[lo]
        if lo > 0:
            prev = self._fifo[lo - 1]
            if abs(prev.timestamp - target_time) < abs(best.timestamp - target_time):
                best = prev

        self._current_output = best.frame
        return self._current_output

    def _draw_overlay(
        self, frame: np.ndarray,
        delay_s: float, latency_ms: int, effective_speed: float,
        auto_speed: bool, bpm: float, tempo_offset: float,
    ) -> np.ndarray:
        """Draw buffer fill indicator, speed, and stats."""
        H, W = frame.shape[:2]
        out = frame.copy()

        # ── Fill bar ──────────────────────────────────────────────────
        bar_h = 8
        bar_margin = 6
        bar_y = H - bar_h - bar_margin
        bar_x = bar_margin
        bar_w = W - bar_margin * 2

        fill = self._buffer_fill(delay_s) if delay_s > 0 else (1.0 if self._fifo else 0.0)

        # Background
        overlay = out.copy()
        cv2.rectangle(overlay, (bar_x - 1, bar_y - 1),
                      (bar_x + bar_w + 1, bar_y + bar_h + 1), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

        # Fill color: red → orange → green → cyan
        fill_w = max(0, int(fill * bar_w))
        if fill < 0.25:
            r, g, b = 220, 50, 50
        elif fill < 0.5:
            t = (fill - 0.25) / 0.25
            r = 220
            g = int(50 + 130 * t)
            b = 50
        elif fill < 0.75:
            t = (fill - 0.5) / 0.25
            r = int(220 * (1 - t) + 50 * t)
            g = int(180 + 30 * t)
            b = int(50 + 50 * t)
        else:
            r, g, b = 50, 210, 180

        if fill_w > 0:
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          (b, g, r), -1)  # BGR

        # Border
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), 1)

        # ── Text ──────────────────────────────────────────────────────
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        thick = 1
        text_y = bar_y - 5
        text_col = (220, 220, 220)

        # Left: delay + speed + BPM
        if latency_ms <= 0:
            label = "PASSTHROUGH"
        else:
            speed_label = f"{effective_speed:.2f}x"
            if auto_speed:
                speed_label = f"AUTO {speed_label}"

            bpm_label = f"{bpm:.0f}bpm"
            if abs(tempo_offset) > 0.1:
                sign = "+" if tempo_offset > 0 else ""
                bpm_label += f" ({sign}{tempo_offset:.0f}%)"

            label = f"DELAY {latency_ms}ms  |  {speed_label}  |  {bpm_label}"

        cv2.putText(out, label, (bar_x, text_y), font, scale, text_col, thick, cv2.LINE_AA)

        # Right: stats
        stats = f"{self._input_fps:.1f} fps  |  {len(self._fifo)}f  |  {fill*100:.0f}%"
        sz = cv2.getTextSize(stats, font, scale, thick)[0]
        cv2.putText(out, stats, (bar_x + bar_w - sz[0], text_y),
                    font, scale, text_col, thick, cv2.LINE_AA)

        # Hold indicator
        if self._hold_active:
            hold_text = "HOLD"
            hsz = cv2.getTextSize(hold_text, font, 0.7, 2)[0]
            hx = (W - hsz[0]) // 2
            hy = (H - hsz[1]) // 2
            cv2.putText(out, hold_text, (hx + 2, hy + 2), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, hold_text, (hx, hy), font, 0.7, (0, 80, 255), 2, cv2.LINE_AA)

        return out
