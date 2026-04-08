"""
pipelines/
Pipeline registry for MM-ContextFold.
"""

_PIPELINE_REGISTRY = {}


def register_pipeline(name: str):
    def decorator(cls):
        _PIPELINE_REGISTRY[name] = cls
        return cls
    return decorator


def get_pipeline(name: str):
    if name not in _PIPELINE_REGISTRY:
        _load_pipelines()

    if name not in _PIPELINE_REGISTRY:
        available = list(_PIPELINE_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline: {name}. Available: {available}")

    return _PIPELINE_REGISTRY[name]


def list_pipelines():
    _load_pipelines()
    return list(_PIPELINE_REGISTRY.keys())


def _load_pipelines():
    import sys

    try:
        from . import pipeline_mmcontextfold
    except ImportError as e:
        print(f"Warning: Failed to load pipeline_mmcontextfold: {e}", file=sys.stderr)
