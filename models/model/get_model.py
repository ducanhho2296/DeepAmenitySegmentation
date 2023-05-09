import segmentation_models_pytorch as smp

def get_unet_model(device, in_channels, num_classes):
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes + 1 # output of model includings batch_size + number of classes
    ).to(device)
    return model

def get_deeplabv3plus_model(device, in_channels, num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes + 1
    ).to(device)
    return model