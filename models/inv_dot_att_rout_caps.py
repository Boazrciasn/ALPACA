#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import torch.nn as nn
import torch.nn.functional as F
import torch

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
'''Capsule in PyTorch
TBD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


#### Simple Backbone ####
class simple_backbone(nn.Module):
    def __init__(self, cl_input_channels, cl_num_filters, cl_filter_size,
                 cl_stride, cl_padding):
        super(simple_backbone, self).__init__()
        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels,
                      out_channels=cl_num_filters,
                      kernel_size=cl_filter_size,
                      stride=cl_stride,
                      padding=cl_padding),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.pre_caps(x)  # x is an image
        return out

    #### ResNet Backbone ####


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_backbone(nn.Module):
    def __init__(self, cl_input_channels, cl_num_filters,
                 cl_stride):
        super(resnet_backbone, self).__init__()
        self.in_planes = 64

        def _make_layer(block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            _make_layer(block=BasicBlock, planes=64, num_blocks=3, stride=1),  # num_blocks=2 or 3
            _make_layer(block=BasicBlock, planes=cl_num_filters, num_blocks=4, stride=cl_stride),  # num_blocks=2 or 4
        )

    def forward(self, x):
        out = self.pre_caps(x)  # x is an image
        return out

    #### Capsule Layer ####


class CapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """

    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, matrix_pose, dp):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.matrix_pose = matrix_pose

        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(out_n_capsules / (self.sqrt_d * in_n_capsules))
            self.w = nn.Parameter(self.weight_init_const * \
                                  torch.randn(in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules))

        else:
            self.weight_init_const = np.sqrt(out_n_capsules / (in_d_capsules * in_n_capsules))
            self.w = nn.Parameter(self.weight_init_const * \
                                  torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules))
        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            weight_init_const={}, dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
            self.weight_init_const, self.dropout_rate
        )

    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        if len(input.shape) == 5:
            input = input.permute(0, 4, 1, 2, 3)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
            input = input.permute(0, 2, 1)

        if self.matrix_pose:
            w = self.w  # nxdm
            _input = input.view(input.shape[0], input.shape[1], self.sqrt_d, self.sqrt_d)  # bnax
        else:
            w = self.w

        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules, self.out_n_capsules).type_as(input)
            query_key = F.softmax(query_key, dim=1)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('nm, bnax, nxdm->bmad', query_key, _input, w)
            else:
                next_capsule_value = torch.einsum('nm, bna, namd->bmd', query_key, input, w)
        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0],
                                                             next_capsule_value.shape[1], self.sqrt_d, self.sqrt_d)
                _query_key = torch.einsum('bnax, nxdm, bmad->bnm', _input, w, next_capsule_value)
            else:
                _query_key = torch.einsum('bna, namd, bmd->bnm', input, w, next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=2)
            query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) + 1e-10)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('bnm, bnax, nxdm->bmad', query_key, _input,
                                                  w)
            else:
                next_capsule_value = torch.einsum('bnm, bna, namd->bmd', query_key, input,
                                                  w)

        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0],
                                                             next_capsule_value.shape[1], self.out_d_capsules)
                next_capsule_value = self.nonlinear_act(next_capsule_value)
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value


class CapsuleCONV(nn.Module):
    r"""Applies as a capsule convolutional layer.
    TBD
    """

    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules,
                 kernel_size, stride, matrix_pose, dp, coordinate_add=False):
        super(CapsuleCONV, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.matrix_pose = matrix_pose
        self.coordinate_add = coordinate_add

        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(out_n_capsules / (self.sqrt_d * in_n_capsules * kernel_size * kernel_size))
            self.w = nn.Parameter(self.weight_init_const * torch.randn(kernel_size, kernel_size,
                                                                       in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules))
        else:
            self.weight_init_const = np.sqrt(out_n_capsules / (in_d_capsules * in_n_capsules * kernel_size * kernel_size))
            self.w = nn.Parameter(self.weight_init_const * torch.randn(kernel_size, kernel_size,
                                                                       in_n_capsules, in_d_capsules, out_n_capsules,
                                                                       out_d_capsules))
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, \
                    kernel_size={}, stride={}, coordinate_add={}, matrix_pose={}, weight_init_const={}, \
                    dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules,
            self.kernel_size, self.stride, self.coordinate_add, self.matrix_pose, self.weight_init_const,
            self.dropout_rate
        )

    def input_expansion(self, input):
        # input has size [batch x num_of_capsule x height x width x  x capsule_dimension]
        unfolded_input = input.unfold(2, size=self.kernel_size, step=self.stride).unfold(3, size=self.kernel_size, step=self.stride)
        unfolded_input = unfolded_input.permute([0, 1, 5, 6, 2, 3, 4])
        # output has size [batch x num_of_capsule x kernel_size x kernel_size x h_out x w_out x capsule_dimension]
        return unfolded_input

    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length
        inputs = self.input_expansion(input)

        if self.matrix_pose:
            w = self.w  # klnxdm
            _inputs = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3], \
                                  inputs.shape[4], inputs.shape[5], self.sqrt_d, self.sqrt_d)  # bnklmhax
        else:
            w = self.w

        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules, self.kernel_size, self.kernel_size,
                                    self.out_n_capsules).type_as(inputs)
            query_key = F.softmax(query_key, dim=3)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('nklm, bnklhwax, klnxdm->bmhwad', query_key,
                                                  _inputs, w)
            else:
                next_capsule_value = torch.einsum('nklm, bnklhwa, klnamd->bmhwd', query_key,
                                                  inputs, w)
        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], \
                                                             next_capsule_value.shape[1], next_capsule_value.shape[2], \
                                                             next_capsule_value.shape[3], self.sqrt_d, self.sqrt_d)
                _query_key = torch.einsum('bnklhwax, klnxdm, bmhwad->bnklmhw', _inputs, w,
                                          next_capsule_value)
            else:
                _query_key = torch.einsum('bnklhwa, klnamd, bmhwd->bnklmhw', inputs, w,
                                          next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=4)
            query_key = query_key / (torch.sum(query_key, dim=4, keepdim=True) + 1e-10)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('bnklmhw, bnklhwax, klnxdm->bmhwad', query_key,
                                                  _inputs, w)
            else:
                next_capsule_value = torch.einsum('bnklmhw, bnklhwa, klnamd->bmhwd', query_key,
                                                  inputs, w)

        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], \
                                                             next_capsule_value.shape[1], next_capsule_value.shape[2], \
                                                             next_capsule_value.shape[3], self.out_d_capsules)
                next_capsule_value = self.nonlinear_act(next_capsule_value)
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)

        return next_capsule_value


# Capsule model
class CapsModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True):

        super(CapsModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing

        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing  # >3 may cause slow converging

        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'],
                                            params['backbone']['stride'],
                                            params['backbone']['padding'])
        elif backbone == 'resnet':
            self.pre_caps = resnet_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['stride'])

        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                  out_channels=params['primary_capsules']['num_caps'] * \
                                               params['primary_capsules']['caps_dim'],
                                  kernel_size=params['primary_capsules']['kernel_size'],
                                  stride=params['primary_capsules']['stride'],
                                  padding=params['primary_capsules']['padding'],
                                  bias=False)

        # self.pc_layer = nn.Sequential()

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])

        ## Main Capsule Layers
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i == 0 else \
                    params['capsules'][i - 1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i == 0 else \
                    params['capsules'][i - 1]['caps_dim']
                self.capsule_layers.append(
                    CapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps,
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'],
                                stride=params['capsules'][i]['stride'],
                                matrix_pose=params['capsules'][i]['matrix_pose'],
                                dp=dp,
                                coordinate_add=False
                                )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] * \
                                params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i - 1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i - 1]['num_caps']
                    in_d_caps = params['capsules'][i - 1]['caps_dim']
                elif params['capsules'][i - 1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i - 1]['num_caps'] * params['capsules'][i - 1]['out_img_size'] * \
                                params['capsules'][i - 1]['out_img_size']
                    in_d_caps = params['capsules'][i - 1]['caps_dim']
                self.capsule_layers.append(
                    CapsuleFC(in_n_capsules=in_n_caps,
                              in_d_capsules=in_d_caps,
                              out_n_capsules=params['capsules'][i]['num_caps'],
                              out_d_capsules=params['capsules'][i]['caps_dim'],
                              matrix_pose=params['capsules'][i]['matrix_pose'],
                              dp=dp
                              )
                )

        ## Class Capsule Layer
        if not len(params['capsules']) == 0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] * \
                            params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] * \
                        params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            CapsuleFC(in_n_capsules=in_n_caps,
                      in_d_capsules=in_d_caps,
                      out_n_capsules=params['class_capsules']['num_caps'],
                      out_d_capsules=params['class_capsules']['caps_dim'],
                      matrix_pose=params['class_capsules']['matrix_pose'],
                      dp=dp
                      )
        )

        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        # self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)

        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c)  # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1)  # 100, 14, 14, 512
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim)  # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4)  # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)  # capsule_utils.squash(u)

        ## Main Capsule Layers
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val)  # get the capsule value for next layer

            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing - 1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n,
                                                          capsule_values[i + 1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing - 1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)

        ## After Capsule
        out = capsule_values[-1]
        out = self.final_fc(out)  # fixed classifier for all capsules
        out = out.squeeze()  # fixed classifier for all capsules
        # out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules

        return out
