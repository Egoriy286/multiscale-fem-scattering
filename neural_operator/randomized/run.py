"""Run the shared CLI with randomized POD and separate local output paths."""

from ..run import main
from . import settings

if __name__ == "__main__":
    main(settings=settings)
