"""
BPM Sync Buffer — Adjustable Latency Buffer for Daydream Scope

Postprocessor that buffers AI-generated frames for smooth, consistent playback.
Instead of bursty chunk output (12 frames at once, then nothing), the buffer
accumulates frames and releases them at a steady rate.

Buffer modes:
  - passthrough: No buffering, frames pass straight through
  - latency:     Adjustable delay (ms) — FIFO + binary search. MIDI-fader friendly.
  - beat:        Beat-locked delay — musical divisions (1/8 to 16 bar) × multiplier

Visual overlay shows buffer fill level, mode, FPS, and delay.

Clock sources (for beat buffer mode):
  - Ableton Link: Networked beat sync with DAWs and other Link-enabled apps
  - MIDI Clock: Standard MIDI timing (24 PPQN) from DJ software, drum machines
  - OSC: Beat position and BPM pushed from external software
  - Internal: Free-running clock at configured BPM (fallback)
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

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


# ─── Buffer mode ───────────────────────────────────────────────────────────

class BufferMode(str, Enum):
    PASSTHROUGH = "passthrough"
    LATENCY = "latency"
    BEAT = "beat"


class BeatDivision(str, Enum):
    """Musical beat subdivisions for rhythm-locked buffer delay."""
    EIGHTH = "1/8"         # Half a beat
    QUARTER = "1/4"        # One beat
    HALF = "1/2"           # Two beats
    ONE_BAR = "1 bar"      # 4 beats
    TWO_BAR = "2 bar"      # 8 beats
    FOUR_BAR = "4 bar"     # 16 beats
    EIGHT_BAR = "8 bar"    # 32 beats
    SIXTEEN_BAR = "16 bar" # 64 beats


# Beat multipliers: how many beats each division represents
_BEAT_MULTIPLIERS = {
    BeatDivision.EIGHTH: 0.5,
    BeatDivision.QUARTER: 1.0,
    BeatDivision.HALF: 2.0,
    BeatDivision.ONE_BAR: 4.0,
    BeatDivision.TWO_BAR: 8.0,
    BeatDivision.FOUR_BAR: 16.0,
    BeatDivision.EIGHT_BAR: 32.0,
    BeatDivision.SIXTEEN_BAR: 64.0,
}


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
    """A frame with wall-clock timestamp for FIFO buffer."""
    frame: np.ndarray        # (H, W, C) uint8
    timestamp: float         # time.monotonic() when received


# ─── Config ────────────────────────────────────────────────────────────────

if _HAS_SCOPE:
    class BpmSyncBufferConfig(BasePipelineConfig):
        """Postprocessor config — latency buffer with visual fill indicator."""
        pipeline_id: str = "bpm_sync_buffer_vjtools"
        pipeline_name: str = "BPM Sync Buffer (VJ.Tools)"
        pipeline_description: str = (
            "Adjustable latency buffer for smooth AI video playback. "
            "Buffers generated frames and releases them at a steady rate. "
            "Beat-locked and millisecond delay modes with visual fill overlay."
        )
        supports_prompts: bool = False
        modified: bool = True
        usage = [UsageType.POSTPROCESSOR]
        modes = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(default=True),
        }

        # --- Performance controls (MIDI-mappable via category="input") ---

        buffer_mode: BufferMode = Field(
            default=BufferMode.LATENCY,
            json_schema_extra=ui_field_config(
                order=0,
                label="Buffer Mode",
                category="input",
            ),
        )

        latency_delay_ms: int = Field(
            default=500,
            ge=0,
            le=60000,
            json_schema_extra=ui_field_config(
                order=1,
                label="Latency (ms)",
                category="input",
            ),
        )

        beat_division: BeatDivision = Field(
            default=BeatDivision.ONE_BAR,
            json_schema_extra=ui_field_config(
                order=2,
                label="Beat Division",
                category="input",
            ),
        )

        beat_multiplier: int = Field(
            default=1,
            ge=1,
            le=16,
            json_schema_extra=ui_field_config(
                order=3,
                label="× Multiplier",
                category="input",
            ),
        )

        show_overlay: bool = Field(
            default=True,
            json_schema_extra=ui_field_config(
                order=4,
                label="Show Overlay",
                category="input",
            ),
        )

        hold: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=5,
                label="HOLD",
                category="input",
            ),
        )

        reset_buffer: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=6,
                label="Reset",
                category="input",
            ),
        )

        # --- Clock config ---

        clock_source: ClockSource = Field(
            default=ClockSource.INTERNAL,
            json_schema_extra=ui_field_config(
                order=7,
                label="Clock Source",
            ),
        )

        clock_bpm: float = Field(
            default=120.0,
            ge=20.0,
            le=999.0,
            json_schema_extra=ui_field_config(
                order=8,
                label="BPM",
                category="input",
            ),
        )

        midi_device: str = Field(
            default="",
            json_schema_extra=ui_field_config(
                order=9,
                label="MIDI Clock Device",
            ),
        )

        osc_beat: float = Field(
            default=0.0,
            ge=0.0,
            json_schema_extra=ui_field_config(
                order=10,
                label="OSC Beat",
                category="input",
            ),
        )
else:
    class BpmSyncBufferConfig:
        """Standalone config for testing outside Scope."""
        def __init__(self, **kwargs):
            self.pipeline_id = kwargs.get("pipeline_id", "bpm_sync_buffer_vjtools")
            self.buffer_mode = kwargs.get("buffer_mode", "latency")
            self.latency_delay_ms = kwargs.get("latency_delay_ms", 500)
            self.beat_division = kwargs.get("beat_division", "1 bar")
            self.beat_multiplier = kwargs.get("beat_multiplier", 1)
            self.show_overlay = kwargs.get("show_overlay", True)
            self.hold = kwargs.get("hold", False)
            self.reset_buffer = kwargs.get("reset_buffer", False)
            self.clock_source = kwargs.get("clock_source", "internal")
            self.clock_bpm = kwargs.get("clock_bpm", 120.0)
            self.midi_device = kwargs.get("midi_device", "")
            self.osc_beat = kwargs.get("osc_beat", 0.0)


# ─── Postprocessor Pipeline ───────────────────────────────────────────────

class BpmSyncBufferPostprocessor(Pipeline):
    """
    Adjustable latency buffer for smooth AI video playback.

    Accumulates AI-generated frames in a wall-clock FIFO and releases them
    at a configurable delay. Binary-searches the FIFO for the closest frame
    to the target playback time.

    This smooths out bursty chunk-based generation (12 frames at once, then
    nothing) into steady-framerate output.

    Visual overlay shows:
      - Fill bar: red → yellow → green → cyan as buffer fills
      - Current mode and delay
      - Input FPS and buffer depth
      - HOLD indicator when frozen
    """

    MAX_FIFO_FRAMES = 1800  # ~60s at 30fps
    FALLBACK_BPM = 120.0

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
        self._current_output: Optional[np.ndarray] = None

        # Hold
        self._hold_active: bool = False
        self._hold_target_time: float = 0.0

        # FPS tracking
        self._input_count = 0
        self._output_count = 0
        self._fps_timestamps: list[float] = []
        self._input_fps = 0.0

        logger.info(f"[Buffer] Initialized (clock={source.value}, delay={getattr(config, 'latency_delay_ms', 500)}ms)")

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

        # --- Read runtime params ---
        mode = str(kwargs.get("buffer_mode", getattr(self.config, "buffer_mode", "latency")))
        latency_ms = int(kwargs.get("latency_delay_ms", getattr(self.config, "latency_delay_ms", 500)))
        beat_div_str = str(kwargs.get("beat_division", getattr(self.config, "beat_division", "1 bar")))
        beat_mult = int(kwargs.get("beat_multiplier", getattr(self.config, "beat_multiplier", 1)))
        show_overlay = kwargs.get("show_overlay", getattr(self.config, "show_overlay", True))
        reset = kwargs.get("reset_buffer", getattr(self.config, "reset_buffer", False))
        hold = kwargs.get("hold", getattr(self.config, "hold", False))

        # --- Clock updates ---
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

        # --- Reset ---
        if reset:
            self._fifo.clear()
            self._current_output = None
            self._hold_active = False
            logger.info("[Buffer] Reset")

        # --- Hold ---
        if hold and not self._hold_active:
            self._hold_active = True
            delay_s = self._delay_seconds(mode, latency_ms, beat_div_str, beat_mult)
            self._hold_target_time = time.monotonic() - delay_s
        elif not hold and self._hold_active:
            self._hold_active = False

        # --- Ingest frames ---
        if isinstance(video, list):
            frames = torch.cat(video, dim=0).float()
        else:
            frames = video.float() if video.dim() == 4 else video.unsqueeze(0).float()

        if frames.max() <= 1.0:
            frames = frames * 255.0

        F, H, W, C = frames.shape
        now = time.monotonic()

        # Track input FPS
        self._input_count += F
        self._fps_timestamps.append(now)
        self._fps_timestamps = [t for t in self._fps_timestamps if now - t < 3.0]
        if len(self._fps_timestamps) > 1:
            span = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if span > 0:
                self._input_fps = (len(self._fps_timestamps) - 1) / span

        # Add to FIFO
        for f_idx in range(F):
            frame_np = frames[f_idx].cpu().numpy().astype(np.uint8)
            self._fifo.append(_BufferedFrame(frame=frame_np, timestamp=now))

        while len(self._fifo) > self.MAX_FIFO_FRAMES:
            self._fifo.pop(0)

        # --- Select output frame ---
        delay_s = self._delay_seconds(mode, latency_ms, beat_div_str, beat_mult)
        if mode == "passthrough" or delay_s <= 0:
            # Zero delay = passthrough (latency fader all the way down)
            output = self._fifo[-1].frame if self._fifo else np.zeros((H, W, C), dtype=np.uint8)
        else:
            output = self._pick_delayed(delay_s)

        if output is None:
            output = np.zeros((H, W, C), dtype=np.uint8)

        self._output_count += 1

        # --- Overlay ---
        if show_overlay:
            output = self._draw_overlay(output, mode, delay_s, latency_ms,
                                         beat_div_str, beat_mult)

        out_tensor = torch.from_numpy(output).float().unsqueeze(0) / 255.0
        return {"video": out_tensor}

    # ── Helpers ─────────────────────────────────────────────────────────

    def _delay_seconds(self, mode: str, latency_ms: int,
                        beat_div_str: str = "1 bar", beat_mult: int = 1) -> float:
        if mode == "beat":
            bpm = self._clock.tempo or self.FALLBACK_BPM
            # Resolve division → base beats
            try:
                division = BeatDivision(beat_div_str)
            except ValueError:
                division = BeatDivision.ONE_BAR
            base_beats = _BEAT_MULTIPLIERS.get(division, 4.0)
            total_beats = base_beats * beat_mult
            return total_beats * 60.0 / bpm
        elif mode == "latency":
            return latency_ms / 1000.0
        return 0.0

    def _pick_delayed(self, delay_s: float) -> Optional[np.ndarray]:
        if not self._fifo:
            return self._current_output

        if self._hold_active and self._hold_target_time > 0:
            target = self._hold_target_time
        else:
            target = time.monotonic() - delay_s

        # Evict old
        cutoff = target - 2.0
        while len(self._fifo) > 1 and self._fifo[0].timestamp < cutoff:
            self._fifo.pop(0)

        if not self._fifo:
            return self._current_output

        # Binary search
        lo, hi = 0, len(self._fifo) - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if self._fifo[mid].timestamp < target:
                lo = mid + 1
            else:
                hi = mid
        best = self._fifo[lo]
        if lo > 0:
            prev = self._fifo[lo - 1]
            if abs(prev.timestamp - target) < abs(best.timestamp - target):
                best = prev

        self._current_output = best.frame
        return self._current_output

    def _draw_overlay(
        self, frame: np.ndarray, mode: str,
        delay_s: float, latency_ms: int,
        beat_div_str: str = "1 bar", beat_mult: int = 1,
    ) -> np.ndarray:
        """Draw buffer fill indicator and stats."""
        H, W = frame.shape[:2]
        out = frame.copy()

        # --- Fill bar ---
        bar_h = 8
        bar_margin = 6
        bar_y = H - bar_h - bar_margin
        bar_x = bar_margin
        bar_w = W - bar_margin * 2

        # Calculate fill
        if delay_s > 0 and self._fifo:
            oldest = self._fifo[0].timestamp
            newest = self._fifo[-1].timestamp
            buffered_s = newest - oldest
            fill = min(1.0, buffered_s / delay_s) if delay_s > 0 else 1.0
        else:
            fill = 0.0 if not self._fifo else 1.0

        # Background
        overlay = out.copy()
        cv2.rectangle(overlay, (bar_x - 1, bar_y - 1),
                      (bar_x + bar_w + 1, bar_y + bar_h + 1), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

        # Fill color: red → orange → green → cyan
        fill_w = max(0, int(fill * bar_w))
        if fill < 0.25:
            # Red
            r, g, b = 220, 50, 50
        elif fill < 0.5:
            # Orange
            t = (fill - 0.25) / 0.25
            r = int(220 + (220 - 220) * t)
            g = int(50 + (180 - 50) * t)
            b = 50
        elif fill < 0.75:
            # Green
            t = (fill - 0.5) / 0.25
            r = int(220 * (1 - t) + 50 * t)
            g = int(180 + (210 - 180) * t)
            b = int(50 + (50) * t)
        else:
            # Cyan-ish green
            r, g, b = 50, 210, 180

        if fill_w > 0:
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          (b, g, r), -1)  # OpenCV is BGR

        # Border
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), 1)

        # --- Text ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        thick = 1
        text_y = bar_y - 5
        text_col = (220, 220, 220)

        # Left: mode + delay
        if mode == "beat":
            bpm = self._clock.tempo or self.FALLBACK_BPM
            div_label = beat_div_str
            if beat_mult > 1:
                label = f"BEAT {beat_mult}x {div_label}  ({delay_s*1000:.0f}ms @ {bpm:.0f}bpm)"
            else:
                label = f"BEAT {div_label}  ({delay_s*1000:.0f}ms @ {bpm:.0f}bpm)"
        elif mode == "latency":
            label = f"DELAY {latency_ms}ms" if latency_ms > 0 else "PASSTHROUGH"
        else:
            label = "PASSTHROUGH"

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
            # Shadow
            cv2.putText(out, hold_text, (hx + 2, hy + 2), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, hold_text, (hx, hy), font, 0.7, (0, 80, 255), 2, cv2.LINE_AA)

        return out
