# from .functions.deform_conv import deform_conv, modulated_deform_conv
# from .functions.deform_pool import deform_roi_pooling
# from .modules.deform_conv import (DeformConv, ModulatedDeformConv,
#                                   DeformConvPack, ModulatedDeformConvPack)
# from .modules.deform_pool import (DeformRoIPooling, DeformRoIPoolingPack,
#                                   ModulatedDeformRoIPoolingPack)

# __all__ = [
#     'DeformConv', 'DeformConvPack', 'ModulatedDeformConv',
#     'ModulatedDeformConvPack', 'DeformRoIPooling', 'DeformRoIPoolingPack',
#     'ModulatedDeformRoIPoolingPack', 'deform_conv', 'modulated_deform_conv',
#     'deform_roi_pooling'
# ]

import os

# Add torch's DLL directory to ensure deform_conv_cuda loads correctly
os.add_dll_directory(r"C:\Users\User\Downloads\Repos\DB\db_env\Lib\site-packages\torch\lib")

from .functions.deform_conv import deform_conv, modulated_deform_conv
from .functions.deform_pool import deform_roi_pooling
from .modules.deform_conv import (DeformConv, ModulatedDeformConv,
                                  DeformConvPack, ModulatedDeformConvPack)
from .modules.deform_pool import (DeformRoIPooling, DeformRoIPoolingPack,
                                  ModulatedDeformRoIPoolingPack)

__all__ = [
    'DeformConv', 'DeformConvPack', 'ModulatedDeformConv',
    'ModulatedDeformConvPack', 'DeformRoIPooling', 'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack', 'deform_conv', 'modulated_deform_conv',
    'deform_roi_pooling'
]