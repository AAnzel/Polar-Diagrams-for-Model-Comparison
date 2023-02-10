# Read version from installed package
from importlib.metadata import version
__version__ = version(__name__)

__all__ = ['df_calculate_td_properties', 'df_calculate_mid_properties',
           'df_calculate_all_properties', 'chart_create_taylor_diagram',
           'chart_create_mi_diagram', 'chart_create_all_diagrams']

# Populate package namespace
from .polar_diagrams import df_calculate_td_properties  # noqa
from .polar_diagrams import df_calculate_mid_properties  # noqa
from .polar_diagrams import df_calculate_all_properties  # noqa
from .polar_diagrams import chart_create_taylor_diagram  # noqa
from .polar_diagrams import chart_create_mi_diagram  # noqa
from .polar_diagrams import chart_create_all_diagrams  # noqa
