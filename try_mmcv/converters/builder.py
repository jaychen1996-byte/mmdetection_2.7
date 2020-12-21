from mmcv.utils import Registry

# create a registry for converters
CONVERTERS = Registry('converter')


# create a build function
def build_converter(cfg, *args, **kwargs):
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in CONVERTERS:
        raise KeyError(f'Unrecognized task type {converter_type}')
    else:
        converter_cls = CONVERTERS.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter
