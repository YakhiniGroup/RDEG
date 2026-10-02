"""Location of separately supplied manuscript-analysis inputs."""
import os
from pathlib import Path


def analysis_root():
    """Use an external private archive without copying its data into Git."""
    return Path(os.environ.get(
        'RDEG_ANALYSIS_ROOT', Path(__file__).resolve().parents[1]
    )).expanduser().resolve()
