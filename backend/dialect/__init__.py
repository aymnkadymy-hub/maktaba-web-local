from .dialect_processor import dialectize, get_system_prompt_block, get_response_prefix
from .marbert_detector  import is_iraqi_dialect, preload as marbert_preload

# Public re-exports — these names are the package's API (imported by
# server_backend / chat_api); __all__ marks them as intentional, not dead.
__all__ = [
    "dialectize", "get_system_prompt_block", "get_response_prefix",
    "is_iraqi_dialect", "marbert_preload",
]
