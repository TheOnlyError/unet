from keras_applications import get_submodules_from_kwargs

from src.segmentation_models import Backbones
from src.segmentation_models.models._utils import filter_keras_submodules
from src.xnet.builder import build_xnet

backend = None
layers = None
models = None
keras_utils = None

def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False
    return


def Xnet(backbone_name='efficientnetb0',
         input_shape=(None, None, 3),
         input_tensor=None,
         encoder_weights='imagenet',
         freeze_encoder=False,
         skip_connections='default',
         decoder_block_type='upsampling',
         decoder_filters=(256, 128, 64, 32, 16),
         decoder_use_batchnorm=True,
         n_upsample_blocks=5,
         upsample_rates=(2, 2, 2, 2, 2),
         classes=1,
         activation='sigmoid',
         **kwargs):
    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)


    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
    )

    if skip_connections == 'default':
        skip_connections = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_xnet(backbone,
                       classes,
                       skip_connections,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    return model
