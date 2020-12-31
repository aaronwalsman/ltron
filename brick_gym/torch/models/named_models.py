import torchvision.models as torch_models

import segmentation_models_pytorch

import brick_gym.torch.models.resnet as bg_resnet
import brick_gym.torch.models.edges as edges
import brick_gym.torch.models.graph_action as graph_action

def get_fcn_model(name, output_channels):
    if name == 'smp_fpn_rnxt50':
        return segmentation_models_pytorch.FPN(
                encoder_name = 'se_resnext50_32x4d',
                encoder_weights = 'imagenet',
                classes = output_channels,
                activation = None)

def get_graphaction_model(name, classes):
    if name == 'spatial_resnet_18':
        backbone = torch_models.resnet18(pretrained=True)
        bg_resnet.make_spatial_attention_resnet(
                backbone, shape=(256,256), do_spatial_embedding=True)
        bg_resnet.replace_fc(backbone, 512)
        model = graph_action.GraphActionModel(backbone, classes, 512)
        return model

def get_edge_model(name):
    if name == 'simple_edge_512':
        return edges.SimpleEdgeModel(
                512, 512)
    '''
    if name == 'dot_edge_512':
        return edges.DotProductEdgeModel(
                512, 512)
    '''
