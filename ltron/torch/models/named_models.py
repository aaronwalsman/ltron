import re
import torch
import torchvision.models as torch_models

import segmentation_models_pytorch

import ltron.torch.models.resnet as bg_resnet
import ltron.torch.models.edge as edge
#import ltron.torch.models.graph as graph
import ltron.torch.models.unet as unet
from ltron.torch.models.graph_step import SlimGraphStepModel, GraphStepModel
from ltron.torch.models.mlp import LinearStack, Conv2dStack
from ltron.torch.models.espnet import ESPNet
from ltron.torch.models.eespnet_seg import EESPNet_Seg
from ltron.torch.models.simple_fcn import SimpleFCN

class UnknownModelError(Exception):
    pass

def backbone_feature_dim(name):
    if 'resnet18' in name:
        return 512
    elif 'resnet34' in name:
        return 512
    elif 'resnet50' in name:
        return 2048
    
    raise UnknownModelError('Unknown backbone: %s'%name)

def named_resnet(name, input_channels=None, num_classes=None):
    pretrained = 'pretrained' in name
    if 'resnet18' in name:
        model = torch_models.resnet18(pretrained=pretrained)
    elif 'resnet34' in name:
        model = torch_models.resnet34(pretrained=pretrained)
    elif 'resnet50' in name:
        model = torch_models.resnet50(pretrained=pretrained)
    if input_channels is not None:
        bg_resnet.replace_conv1(model, input_channels)
    if num_classes is not None:
        bg_resnet.replace_fc(num_classes)
    
    return model

def named_backbone(name, input_channels=None):
    if 'resnet' in name:
        model = named_resnet(name, input_channels)
        backbone = bg_resnet.ResnetBackbone(model)
        return backbone
    
    raise UnknownModelError('Unknown backbone: %s'%name)

def named_single_feature_backbone(name, shape, input_channels=None):
    if 'resnet' in name:
        model = named_resnet(name, input_channels)
        if 'attention' in name:
            bg_resnet.make_spatial_attention_resnet(
                    model, shape, do_spatial_embedding=True)
            bg_resnet.replace_fc(model, backbone_feature_dim(name))
        else:
            raise UnknownModelError(
                    'Unknown setting for single feature backbone: %s'%name)
        return model
    
    raise UnknownModelError('Unknown single feature backbone: %s'%name)

# TODO: Cleanup!
def named_fcn_backbone(name, output_channels):
    if name == 'smp_fpn_rnxt50':
        return segmentation_models_pytorch.FPN(
                encoder_name = 'se_resnext50_32x4d',
                encoder_weights = 'imagenet',
                classes = output_channels,
                activation = None)
    elif name == 'smp_fpn_r50':
        return segmentation_models_pytorch.FPN(
                encoder_name = 'resnet50',
                encoder_weights = 'imagenet',
                classes = output_channels,
                activation = None)
    elif name == 'smp_fpn_r34':
        return segmentation_models_pytorch.FPN(
                encoder_name = 'resnet34',
                encoder_weights = 'imagenet',
                classes = output_channels,
                activation = None)
    elif name == 'smp_fpn_r18':
        return segmentation_models_pytorch.FPN(
                encoder_name = 'resnet18',
                encoder_weights = 'imagenet',
                classes = output_channels,
                activation = None)
    
    elif name == 'espnet':
        return ESPNet(classes=output_channels)
    
    elif name == 'eespnet':
        return EESPNet_Seg(classes=output_channels,
                pretrained='/media/awalsman/data_drive/brick-gym/ltron/torch/models/espnetv2_s_1.0.pth')
    elif name == 'simple':
        return SimpleFCN(decoder_channels=output_channels)
    
    if 'unet' in name:
        if 'coord' in name:
            coord_conv = True
        else:
            coord_conv = False
        feature_dim = 32 # 64?
        return unet.UNet(3, feature_dim, output_channels, coord_conv)
    
    raise UnknownModelError('Unknown fcn backbone: %s'%name)

'''
def named_edge_model(name, input_dim):
    if name == 'feature_difference_512':
        return edge.FeatureDifferenceEdgeModel(input_dim, 512)
    
    raise UnknownModelError('Unknown edge model: %s'%name)
'''

def named_dense_edge_model(name, input_dim):
    if name == 'squared_difference':
        return edge.DenseEdgeModel(
                input_dim,
                pre_compare_layers=3,
                post_compare_layers=3,
                compare_mode = 'squared_difference')
    else:
        raise NotImplementedError

def named_edge_model(name, input_dim):
    if name == 'default':
        return edge.EdgeModel(
                input_dim,
                pre_compare_layers=3,
                post_compare_layers=3,
                compare_mode='add')
    elif name == 'subtract':
        return edge.EdgeModel(
                input_dim,
                pre_compare_layers=3,
                post_compare_layers=3,
                compare_mode='subtract')
    
    elif name == 'squared_difference':
        return edge.EdgeModel(
                input_dim,
                pre_compare_layers=3,
                post_compare_layers=3,
                compare_mode='squared_difference')

def named_graph_model(
        name, backbone_name, edge_model_name, node_classes, shape):
    # TODO: Better name for this once we have other models
    if name == 'first_try':
        backbone_dim = backbone_feature_dim(backbone_name)
        backbone_model = named_single_feature_backbone(
                backbone_name, shape, input_channels=4)
        node_model = torch.nn.Linear(backbone_dim, node_classes)
        action_model = torch.nn.Linear(backbone_dim, 1)
        edge_model = named_edge_model(edge_model_name, backbone_dim)
        return graph.GraphModel(
                backbone_model, node_model, edge_model, action_model)
        
    raise UnknownModelError('Unknown graph model: %s'%name)

def named_graph_pool_model(
        name,
        backbone_name,
        edge_model_name,
        node_classes,
        output_channels=32,
        shape=(256,256)):
    if name == 'second_try':
        backbone_model = named_fcn_backbone(backbone_name, output_channels)
        node_model = torch.nn.Linear(output_channels, node_classes)
        action_model = torch.nn.Linear(output_channels, 1)
        edge_model = named_edge_model(edge_model_name, output_channels)
        return graph.GraphPoolModel(
                backbone_model,
                None,
                node_model,
                edge_model,
                action_model,
                shape,
                output_channels)
'''
def named_graph_step_model(
        name,
        backbone_name,
        brick_types,
        brick_vector_channels=256,
        shape=(256,256)):
    if name == 'ground_truth_segmentation':
        backbone = named_fcn_backbone(backbone_name, brick_vector_channels)
        brick_feature_backbone = BrickFeatureBackbone(
                backbone, brick_feature_model)
        heads = {
                'brick_features' : torch.nn.Identity(),
                'brick_type' : LinearReluStack(
                        3,
                        brick_vector_channels,
                        brick_vector_channels,
                        brick_types),
                'add_to_graph' : LinearReluStack(
                        3,
                        brick_vector_channels,
                        brick_vector_channels,
                        1)
                'hide_action' : LinearReluStack(
                        3,
                        brick_vector_channels,
                        brick_vector_channels,
                        1)
        }
'''
def named_slim_graph_step_model(
        name,
        backbone_name,
        decoder_channels,
        num_classes,
        input_resolution,
        viewpoint_head=False):
    if name == 'oth_try':
        if backbone_name == 'simple':
            output_resolution = (
                    input_resolution[0]//4, input_resolution[1]//4)
        else:
            raise NotImplementedError
        if viewpoint_head:
            single_heads = {
                'viewpoint' : torch.nn.Linear(2048, 4)
            }
        else:
            single_heads = {}
        return SlimGraphStepModel(
                backbone = named_fcn_backbone(backbone_name, decoder_channels),
                add_spatial_embedding=True,
                decoder_channels = decoder_channels,
                output_resolution = output_resolution,
                dense_heads = {
                    'x' : torch.nn.Identity(),
                    'instance_label' : Conv2dStack(
                        3, decoder_channels, decoder_channels, num_classes),
                    'instance_confidence' : Conv2dStack(
                        3, decoder_channels, decoder_channels, 1),
                    'edge_confidence' : Conv2dStack(
                        3, decoder_channels, decoder_channels, 1)},
                single_heads = single_heads
        )
    else:
        raise NotImplementedError

def named_graph_step_model(
        name,
        backbone_name,
        decoder_channels,
        num_classes,
        input_resolution,
        viewpoint_head=False,
        removability_head=False):
    if name == 'nth_try':
        #encoder = 'smp_fpn_r18' #'smp_fpn_rnxt50'
        if backbone_name == 'simple':
            output_resolution = (
                    input_resolution[0]//4, input_resolution[1]//4)
        else:
            output_resolution = input_resolution
        if viewpoint_head:
            single_heads = {
                'viewpoint' : torch.nn.Linear(2048, 7)
            }
        else:
            single_heads = {}
        dense_heads = {
            'x' : torch.nn.Identity(),
            'instance_label' : Conv2dStack(
                    3, decoder_channels, decoder_channels, num_classes),
            'hide_action' : Conv2dStack(
                    3, decoder_channels, decoder_channels, 1)
        }
        if removability_head:
            dense_heads['removability'] = Conv2dStack(
                    3, decoder_channels, decoder_channels, 1)
        
        return GraphStepModel(
                backbone = named_fcn_backbone(backbone_name, decoder_channels),
                score_model = Conv2dStack(
                        3, decoder_channels, decoder_channels, 1),
                segmentation_model = None,
                add_spatial_embedding = True,
                add_viewpoint_embedding = viewpoint_head,
                decoder_channels = decoder_channels,
                output_resolution = output_resolution,
                dense_heads = dense_heads,
                single_heads = single_heads,
        )
    elif name == 'center_voting':
        if backbone_name == 'simple':
            output_resolution = (
                    input_resolution[0]//4, input_resolution[1]//4)
        else:
            output_resolution = input_resolution
        if viewpoint_head:
            single_heads = {
                'viewpoint' : torch.nn.Linear(2048, 4)
            }
        else:
            single_heads = {}
        return GraphStepModel(
                backbone = named_fcn_backbone(backbone_name, decoder_channels),
                score_model = Conv2dStack(
                        3, decoder_channels, decoder_channels, 1),
                segmentation_model = None,
                add_spatial_embedding = True,
                decoder_channels = decoder_channels,
                output_resolution = output_resolution,
                dense_heads = {
                    'x' : torch.nn.Identity(),
                    'instance_label' : Conv2dStack(
                            3, decoder_channels, decoder_channels, num_classes),
                    'hide_action' : Conv2dStack(
                            3, decoder_channels, decoder_channels, 1),
                    'cluster_center' : Conv2dStack(
                            3, decoder_channels, decoder_channels, 3),
                },
                single_heads = single_heads,
        )
    
    elif name == 'nth_try_nose':
        return GraphStepModel(
                backbone = named_fcn_backbone('smp_fpn_rnxt50', 256),
                score_model = Conv2dStack(3, 256, 256, 1),
                segmentation_model = None,
                add_spatial_embedding = False,
                heads = {
                    'x' : torch.nn.Identity(),
                    'instance_label' : Conv2dStack(3, 256, 256, num_classes),
                    'hide_action' : Conv2dStack(3, 256, 256, 1)
                })
    elif name == 'fcos':
        if backbone_name == 'simple':
            output_resolution = (64, 64)
        else:
            output_resolution = (256, 256)
        return GraphStepModel(
                backbone = named_fcn_backbone(backbone_name, decoder_channels),
                score_model = Conv2dStack(
                        3, decoder_channels, decoder_channels, 1),
                segmentation_model = FCOS_SEGMENTATION_CONVERTER,
                add_spatial_embedding = True,
                decoder_channels = decoder_channels,
                output_resolution = output_resolution,
                heads = {
                    'x' : torch.nn.Identity(),
                    'instance_label' : Covn2dStack(
                            3, decoder_channels, decoder_channels, num_classes),
                    'hide_action' : Conv2dStack(
                            3, decoder_channels, decoder_channels, 1),
                    'fcos_boxes' : FCOS_BOX_HEAD
                })
