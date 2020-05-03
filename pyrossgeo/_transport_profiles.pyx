from pyrossgeo.csimulation cimport DTYPE_t, csimulation, node, cnode, transporter
from pyrossgeo.csimulation import DTYPE

ctypedef DTYPE_t (*transport_profile)(DTYPE_t t)

# https://stackoverflow.com/questions/14124049/is-there-any-type-for-function-in-cython