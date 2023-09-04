from backbones.deeplab import mobilenet


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError