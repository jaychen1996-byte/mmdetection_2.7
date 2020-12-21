import mmcv

"""
MMCV implements registry to manage different modules that share similar functionalities, e.g., backbones, head, and necks, in detectors. 
Most projects in OpenMMLab use registry to manage modules of datasets and models, such as MMDetection, MMDetection3D, MMClassification, MMEditing, etc.
"""

# What is registry?
"""
In MMCV, registry can be regarded as a mapping that maps a class to a string. 
These classes contained by a single registry usually have similar APIs but implement different algorithms or support different datasets. 
With the registry, users can find and instantiate the class through its corresponding string, and use the instantiated module as they want. 
One typical example is the config systems in most OpenMMLab projects, which use the registry to create hooks, runners, models, and datasets, through configs.
"""

# To manage your modules in the codebase by Registry, there are three steps as below.
"""
1.Create an registry
2.Create a build method
3.Use this registry to manage the modules
"""

# A Simple Example
"""
Here we show a simple example of using registry to manage modules in a package. 

You can find more practical examples in OpenMMLab projects.

Assuming we want to implement a series of Dataset Converter for converting different formats of data to the expected data format. 

We create directory as a package named converters. In the package, we first create a file to implement builders, named converters/builder.py, as below:
"""

# from mmcv.utils import Registry
#
# # create a registry for converters
# CONVERTERS = Registry('converter')
#
#
# # create a build function
# def build_converter(cfg, *args, **kwargs):
#     cfg_ = cfg.copy()
#     converter_type = cfg_.pop('type')
#     if converter_type not in CONVERTERS:
#         raise KeyError(f'Unrecognized task type {converter_type}')
#     else:
#         converter_cls = CONVERTERS.get(converter_type)
#
#     converter = converter_cls(*args, **kwargs, **cfg_)
#     return converter

from try_mmcv.converters import build_converter

converter_cfg = dict(type='Converter1', a=1, b=2)
converter = build_converter(converter_cfg)
print(converter.a)
print(converter.b)
print(type(converter))
