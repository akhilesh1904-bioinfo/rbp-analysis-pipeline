"""Sequence-based RNA-binding protein (RBP) binding-site analysis from eCLIP peaks."""

__version__ = "0.5.2"


class PipelineError(RuntimeError):
    """A problem with inputs or data that stops the pipeline with a clear message."""
