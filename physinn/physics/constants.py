"""Physical constants used by the spectroscopy model."""

import math

C = 2.99792458e10  # cm/s
NA = 6.02214129e23
KB = 1.380649e-16  # erg/K
R = NA * KB  # erg/(mol.K)
P0 = 1013.25  # mbar
T0 = 273.15  # K
TREF = 296.0
L0 = 2.6867773e19  # cm^-3
C2 = 1.438776877  # cm.K
SQRT_LN2 = math.sqrt(math.log(2.0))
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

MOLECULE_PARAMS = {
    "CH4": {"M": 16.04, "PL": 15.12},
    "H2O": {"M": 18.01528, "PL": 15.12},
}

