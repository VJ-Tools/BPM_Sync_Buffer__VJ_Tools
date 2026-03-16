"""
BPM Sync Buffer — Scope Plugin Registration

Registers the latency buffer postprocessor with Daydream Scope.
"""

import logging

try:
    from scope.core.plugins.hookspecs import hookimpl
except ImportError:
    def hookimpl(f):
        return f

from .pipeline import BpmSyncBufferPostprocessor

logger = logging.getLogger(__name__)


@hookimpl
def register_pipelines(register):
    """Register BPM Sync Buffer postprocessor with Scope."""
    register(BpmSyncBufferPostprocessor)
    logger.info("[BPM Buffer] Registered postprocessor")
