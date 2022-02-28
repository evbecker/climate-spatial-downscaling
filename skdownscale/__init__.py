from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # noqa: F401
    # package is not installed
    __version__ = '0.0.0'  # Fixme
