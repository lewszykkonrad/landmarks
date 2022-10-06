import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')