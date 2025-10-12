import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    一个统一的卷积网络类，可以通过 'mode' 参数在两种不同配置之间切换。
    - 'vision':  一个较深的网络 (5层)，带有可选的全连接层用于嵌入。
    - 'audio': 一个较浅的网络 (3层)，在最后一层使用自适应池化。
    """
    def __init__(self, channel, im_size=(32,32), mode='vision', num_classes=None):
        super(ConvNet, self).__init__()
        
        # 1. 根据模式设置网络参数
        self.mode = mode
        if self.mode == 'vision':
            # 'ConvNet_v' 的设置
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 5, 'relu', 'instancenorm', 'avgpooling'
            self.sample_layer = None # 稍后定义
        elif self.mode == 'audio':
            # 'ConvNet' 的设置
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
            self.sample_layer = None
        else:
            raise ValueError(f"未知的模式: {mode}. 可选项为 'vision' 或 'audio'。")

        # 2. 构建特征提取层
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        
        # 3. 如果是 'vision' 模式，则创建额外的全连接层
        if self.mode == 'vision':
            num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
            self.sample_layer = nn.Linear(num_feat, 4096)

        self.classifier = None
        if num_classes is not None and num_classes > 0:
            # 计算卷积部分的输出特征维度
            num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
            self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        """
        前向传播仅通过卷积特征提取器。
        这与两个原始类中的 forward 行为一致。
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)

        if self.classifier is not None:
            out = self.classifier(out)
        return out

    def embed(self, x, sample_match=False):
        """
        生成嵌入向量。
        对于 'vision' 模式，可以根据 sample_match 选择是否通过额外的全连接层。
        对于 'audio' 模式，此方法行为与 forward 方法相同。
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        
        # 仅在 vision 模式且 sample_match 为 True 时应用 sample_layer
        if self.mode == 'vision' and sample_match and self.sample_layer is not None:
            out = self.sample_layer(out)
            out = nn.ReLU(inplace=True)(out)
            
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            
            if net_pooling != 'none':
                # 关键区别：'audio' 模式在最后一层使用自适应池化
                if self.mode == 'audio' and d == net_depth - 1:
                    layers += [nn.AdaptiveAvgPool2d((7, 7))]
                    # 注意：自适应池化后尺寸固定为 7x7，但为了通用性，我们仍按常规计算 shape_feat
                    shape_feat[1] = 7
                    shape_feat[2] = 7
                else:
                    layers += [self._get_pooling(net_pooling)]
                    shape_feat[1] //= 2
                    shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat