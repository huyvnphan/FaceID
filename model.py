import os
import torch
from cnn_architecture.squeeze_net import SqueezeNet
from cnn_architecture.shuffle_net import ShuffleNetV2
from cnn_architecture.res_net import ResNet


def get_model(arch):
    if arch == "squeeze_net":
        return SqueezeNet()
    elif arch == "shuffle_net":
        return ShuffleNetV2()
    elif arch == "res_net":
        return ResNet()


def FaceIDModel(arch, pretrained):
    model = get_model(arch)

    if pretrained == 1:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            os.path.join(script_dir, "weights", arch + ".pt"),
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(state_dict)
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model
