import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import time
import json
from nets.imagebind import data
from tqdm import tqdm
import json

def pad_mel_spec(mel, target_len=204):
    cur_len = mel.shape[-1]
    if cur_len >= target_len:
        return mel[..., :target_len]
    else:
        pad_width = target_len - cur_len
        pad_shape = list(mel.shape)
        pad_shape[-1] = pad_width
        return torch.cat([mel, torch.zeros(pad_shape, device=mel.device, dtype=mel.dtype)], dim=-1)

class CombTensorDataset_origin(Dataset):
    def __init__(self, audio, images, labels, args):
        self.args = args
        self.labels = labels.detach() if torch.is_tensor(labels) else torch.tensor(labels)

            
        # 音频
        if args.input_modality in ['a', 'av']:
            if isinstance(audio, torch.Tensor):
                audio_data = audio.detach().float()
            elif isinstance(audio, list):
                if all(isinstance(x, torch.Tensor) for x in audio):
                    audio_data = torch.stack(audio).detach().float()
                elif all(isinstance(x, str) for x in audio):
                    # 路径列表，批量读取
                    audio_tensor_list = [
                        data.load_and_transform_audio_data([audio_path], 'cpu')
                        for audio_path in tqdm(audio, desc="加载音频")
                    ]
                    audio_data = torch.cat(audio_tensor_list, dim=0)  # 按实际shape决定dim
                else:
                    raise RuntimeError(f"audio列表元素类型未知: {type(audio[0])}")
            elif isinstance(audio, str):
                # 单个路径
                audio_data = data.load_and_transform_audio_data([audio], 'cpu')
            else:
                raise RuntimeError(f"audio格式未知, type: {type(audio)}")
            
            # 只做pad/trunc
            if audio_data.shape[-2] == 128 and audio_data.shape[-1] != 204:
                print(f"[Dataset] audio before pad: {audio_data.shape}")
                audio_data = pad_mel_spec(audio_data, 204)
                print(f"[Dataset] audio after pad: {audio_data.shape}")
            self.audio_data = audio_data
        else:
            self.audio_data = None

        # 图像
        if args.input_modality in ['v', 'av']:
            if isinstance(images, torch.Tensor):
                images_data = images.detach().float()
            elif isinstance(images, list):
                if all(isinstance(x, torch.Tensor) for x in images):
                    images_data = torch.stack(images).detach().float()
                elif all(isinstance(x, str) for x in images):
                    images_tensor_list = [
                        data.load_and_transform_vision_data([image_path], 'cpu')
                        for image_path in tqdm(images, desc="加载图像")
                    ]
                    images_data = torch.cat(images_tensor_list, dim=0)
                else:
                    raise RuntimeError(f"images列表元素类型未知: {type(images[0])}")
            elif isinstance(images, str):
                images_data = data.load_and_transform_vision_data([images], 'cpu')
            else:
                raise RuntimeError(f"images格式未知, type: {type(images)}")
            self.frame_data = images_data
        else:
            self.frame_data = None
    def __getitem__(self, idx):
        audio, frame = torch.zeros(1), torch.zeros(1)
        label = self.labels[idx]
        if self.args.input_modality in ['a', 'av']:
            audio = self.audio_data[idx]
        if self.args.input_modality in ['v', 'av']:
            frame = self.frame_data[idx]
        return {'frame': frame, 'audio': audio, 'label': label}

    def __len__(self):
        return self.labels.shape[0]
    
def get_train_dataset_origin(dataset, args):
    
    if dataset == 'VGG_subset':
        data = torch.load('data/train_data/vgg_subset.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10
        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]
        aud_size = (128, 204)
        im_size = (224, 224)
        im_size = [aud_size, im_size]

        # 音频 [N, 1, 128, 56] → [N, 1, 128, 204]
        aud_train = data['audio_train'].detach().float()
        if aud_train.shape[-2] == 128 and aud_train.shape[-1] != 204:
            aud_train = pad_mel_spec(aud_train, 204)

        # 图像 [N, 3, H, W] → [N, 3, 224, 224]
        images_train = data['images_train'].detach().float() / 255.0
        if images_train.shape[-2:] != (224, 224):
            import torch.nn.functional as F
            images_train = F.interpolate(images_train, size=(224, 224), mode='bilinear', align_corners=False)
        # 标准化
        for c in range(3):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]

        labels_train = data['labels_train'].detach().long()

        dst_train = CombTensorDataset_origin(aud_train, images_train, labels_train, args)
        return channel, im_size, num_classes, mean, std, dst_train
    elif dataset == 'AVE':
        # data = torch.load('data/train_data/ave_train.pt', map_location='cpu')
        with open('data/train_data/ave_train.json', 'r') as file:
            data = json.load(file)
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 3
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 204)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        aud_train = []
        images_train = []
        labels_train = []
        for data_item in data:
            aud_train.append(data_item['audio'])
            images_train.append(data_item['frame'])
            labels_train.append(data_item['label'])
        dst_train = CombTensorDataset_origin(aud_train, images_train, labels_train, args)
    elif dataset == 'VGG':
        data_path = 'data/train_data/vggsound/vggsound.pt'
        
        # 检查是否使用流式加载
        use_streaming = getattr(args, 'streaming', False)
        max_samples = getattr(args, 'max_samples', None)
        
        if use_streaming:
            print("使用流式数据加载模式...")
            dst_train = VGGStreamDataset(data_path, args, max_samples=max_samples)
            
            # 设置通道和尺寸信息
            aud_channel = 1
            im_channel = 3
            channel = [aud_channel, im_channel]
            aud_size = (128, 204)
            im_size = (224, 224)
            im_size = [aud_size, im_size]
            mean = [0.425, 0.396, 0.370]
            std = [0.229, 0.224, 0.221]
            num_classes = 309
            
        else:
            # 原始加载方式（一次性加载）
            print("使用一次性数据加载模式...")
            data = torch.load(data_path, map_location='cpu')
            mean = [0.425, 0.396, 0.370]
            std =  [0.229, 0.224, 0.221]
            num_classes = 309

            aud_channel = 1
            im_channel = 3
            channel = [aud_channel, im_channel]

            aud_size = (128, 204)
            im_size = (224, 224)
            im_size = [aud_size, im_size]
            
            # 音频 [N, 1, 128, 56] → [N, 1, 128, 204]
            aud_train = data['audio_train'].detach().float()
            if max_samples:
                aud_train = aud_train[:max_samples]
            if aud_train.shape[-2] == 128 and aud_train.shape[-1] != 204:
                aud_train = pad_mel_spec(aud_train, 204)

            # 图像 [N, 3, H, W] → [N, 3, 224, 224]
            images_train = data['images_train'].detach().float() / 255.0
            if max_samples:
                images_train = images_train[:max_samples]
            if images_train.shape[-2:] != (224, 224):
                import torch.nn.functional as F
                images_train = F.interpolate(images_train, size=(224, 224), mode='bilinear', align_corners=False)
            for c in range(3):
                images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]

            labels_train = data['labels_train'].detach().long()
            if max_samples:
                labels_train = labels_train[:max_samples]

            dst_train = CombTensorDataset_origin(aud_train, images_train, labels_train, args)
    else:
        exit('unknown dataset: %s'%dataset)
    
    return channel, im_size, num_classes, mean, std, dst_train


def get_test_dataset_origin(dataset, args):
    

    if dataset == 'VGG_subset':
        data = torch.load('data/test_data/vgg_subset_test.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10
        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]
        aud_size = (128, 204)
        im_size = (224, 224)
        im_size = [aud_size, im_size]

        # 音频 [N, 1, 128, 56] → [N, 1, 128, 204]
        aud_test = data['audio_test'].detach().float()
        if aud_test.shape[-2] == 128 and aud_test.shape[-1] != 204:
            aud_test = pad_mel_spec(aud_test, 204)

        # 图像 [N, 3, H, W] → [N, 3, 224, 224]
        images_test = data['images_test'].detach().float() / 255.0
        if images_test.shape[-2:] != (224, 224):
            import torch.nn.functional as F
            images_test = F.interpolate(images_test, size=(224, 224), mode='bilinear', align_corners=False)
        for c in range(3):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]

        labels_test = data['labels_test'].detach().long()
        dst_test = CombTensorDataset_origin(aud_test, images_test, labels_test, args)
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=32, shuffle=False, num_workers=args.num_workers)
        return channel, im_size, num_classes, testloader
    elif dataset == 'Music_21':
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 21

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]

        data = torch.load('data/test_data/music_21_test.pt', map_location='cpu')
        #Test
        aud_test = data['audio_test']
        images_test = data['images_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)  

    elif dataset == 'AVE':
        # data = torch.load('data/test_data/ave_test.pt', map_location='cpu')
        with open('data/test_data/ave_test.json', 'r') as file:
            data = json.load(file)
        #common
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 3
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 204)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        aud_test = []
        images_test = []
        labels_test = []
        for data_item in data:
            aud_test.append(data_item['audio'])
            images_test.append(data_item['frame'])
            labels_test.append(data_item['label'])
        

        dst_test = CombTensorDataset_origin(aud_test, images_test, labels_test, args)

    elif dataset == 'VGG':
        data = torch.load('data/test_data/vggsound/vgg_test.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 309

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 204)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # 1. 音频 [N, 1, 128, 56] → [N, 1, 128, 204]
        aud_test = data['audio_test'].detach().float()
        if aud_test.shape[-2] == 128 and aud_test.shape[-1] != 204:
            aud_test = pad_mel_spec(aud_test, 204)

        # 2. 图像 [N, 3, H, W] → [N, 3, 224, 224]
        images_test = data['images_test'].detach().float() / 255.0
        if images_test.shape[-2:] != (224, 224):
            import torch.nn.functional as F
            images_test = F.interpolate(images_test, size=(224, 224), mode='bilinear', align_corners=False)
        for c in range(3):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]

        # 3. 标签
        labels_test = data['labels_test'].detach().long()

        # 4. Dataset和Loader
        dst_test = CombTensorDataset_origin(aud_test, images_test, labels_test, args)

    else:
        exit('unknown dataset: %s'%dataset)
    
    bs = 32
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=bs, shuffle=False, num_workers=args.num_workers)

    return channel, im_size, num_classes, testloader


# class CombTensorDataset(Dataset):
#     def __init__(self, audio, images, labels, args): # images: n x c x h x w tensor
#         if args.input_modality == 'a' or args.input_modality == 'av':
#             self.audio = audio.detach().float()
#         if args.input_modality == 'v' or args.input_modality == 'av':
#             self.images = images.detach().float()
#         self.labels = labels.detach()
#         self.args = args

#     def __getitem__(self, index):
#         audio, frame = torch.zeros(1), torch.zeros(1)
#         label = self.labels[index]
#         if self.args.input_modality == 'a' or self.args.input_modality == 'av':
#             audio = self.audio[index]
#         if self.args.input_modality == 'v' or self.args.input_modality == 'av':
#             frame = self.images[index] 
#         ret_dict = {'frame': frame, 'audio': audio, 'label':label}
#         return ret_dict

#     def __len__(self):
#         return self.labels.shape[0]
    
# # class CombTensorDataset_origin(Dataset):
# #     def __init__(self, audio, images, labels, args): # images: n x c x h x w tensor
# #         if args.input_modality == 'a' or args.input_modality == 'av':
# #             self.audio_paths = audio
# #         if args.input_modality == 'v' or args.input_modality == 'av':
# #             self.images_paths = images
# #         self.labels = labels
# #         self.args = args

# #     def __getitem__(self, index):
# #         audio, frame = torch.zeros(1), torch.zeros(1)
# #         label = self.labels[index]
# #         if self.args.input_modality == 'a' or self.args.input_modality == 'av':
# #             audio_path = [self.audio_paths[index]]
# #         if self.args.input_modality == 'v' or self.args.input_modality == 'av':
# #             frame_path = [self.images_paths[index]]
# #         audio = data.load_and_transform_audio_data(audio_path, 'cpu')  
# #         frame = data.load_and_transform_vision_data(frame_path, 'cpu')
# #         ret_dict = {'frame': frame, 'audio': audio, 'label':label}
# #         return ret_dict

# #     def __len__(self):
# #         return len(self.labels)

    

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import time
import json

def get_test_dataset(dataset, args):
    
    if dataset == 'VGG_subset':
        data = torch.load('data/test_data/vgg_subset_test.pt', map_location='cpu')
        
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10

        aud_channel = 3
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (1, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        #Test
        aud_test = data['audio_test']
        images_test = data['images_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        aud_test = aud_test.mean(dim=2, keepdim=True)
        aud_test = aud_test.repeat(1, 3, 1, 1)
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)

    elif dataset == 'Music_21':
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 21

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]

        data = torch.load('data/test_data/music_21_test.pt', map_location='cpu')
        #Test
        aud_test = data['audio_test']
        images_test = data['images_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)  

    elif dataset == 'AVE':
        data = torch.load('data/test_data/ave_test.pt', map_location='cpu')
        #common
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        #Test
        aud_test = data['audio_test']
        images_test = data['images_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)

    elif dataset == 'VGG':
        data = torch.load('data/test_data/vggsound/vgg_test.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 309

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # Test
        images_test = data['images_test']
        aud_test = data['audio_test']
        labels_test = data['labels_test']
        
        aud_test = aud_test.detach().float()
        images_test = images_test.detach().float() / 255.0
        labels_test = labels_test.detach().long()
        for c in range(channel[1]):
            images_test[:, c] = (images_test[:, c] - mean[c]) / std[c]
        dst_test = CombTensorDataset(aud_test, images_test, labels_test, args)

    else:
        exit('unknown dataset: %s'%dataset)
    
    bs = 32
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=bs, shuffle=False, num_workers=args.num_workers)

    return channel, im_size, num_classes, testloader


def get_train_dataset(dataset, args):
    
    # if dataset == 'VGG_subset':
    #     data = torch.load('data/train_data/vgg_subset_train.pt', map_location='cpu')
        
    #     mean = [0.425, 0.396, 0.370]
    #     std =  [0.229, 0.224, 0.221]
    #     num_classes = 10

    #      # —— 让 VGG_subset 的音频适配 ImageBind ——
    #     aud_channel = 3                      # ImageBind 需要 3-channel
    #     im_channel = 3
    #     channel = [aud_channel, im_channel]

    #     aud_size = (1, 56)                   # H 压缩到 1
    #     im_size = (224, 224)
    #     im_size = [aud_size, im_size]
        
    #     aud_train = data['audio_train']
    #     images_train = data['images_train']
    #     labels_train = data['labels_train']
        
    #     aud_train = aud_train.detach().float()
    #      # ★ 把 [B,1,128,56] → [B,3,1,56] ★
    #     aud_train = aud_train.mean(dim=2, keepdim=True)   # 频率平均，H=1
    #     aud_train = aud_train.repeat(1, 3, 1, 1)          # 复制到 3 通道
    #     images_train = images_train.detach().float() / 255.0
    #     labels_train = labels_train.detach().long()
    #     for c in range(channel[1]):
    #         images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]
    #     dst_train = CombTensorDataset(aud_train, images_train, labels_train, args)
    if dataset == 'VGG_subset':
        data = torch.load('data/train_data/vgg_subset_train.pt', map_location='cpu')
        
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # Load data
        aud_train = data['audio_train']   # (N, 1, 128, 56)
        images_train = data['images_train']  # (N, 3, H, W)
        labels_train = data['labels_train']
        
        aud_train = aud_train.detach().float()
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach().long()

        # 强制resize图片到(224,224)
        if images_train.shape[-2:] != (224, 224):
            images_train = F.interpolate(images_train, size=(224, 224), mode='bilinear', align_corners=False)

        # 如需将音频作为图片输入模型，也要resize
        if aud_train.shape[-2:] != (224, 224):
            aud_train = F.interpolate(aud_train, size=(224, 224), mode='bilinear', align_corners=False)

        # 标准化图片
        for c in range(channel[1]):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]

        dst_train = CombTensorDataset(aud_train, images_train, labels_train, args)

        return channel, im_size, num_classes, mean, std, dst_train
    elif dataset == 'AVE':
        data = torch.load('data/train_data/ave_train.pt', map_location='cpu')
        
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        aud_train = data['audio_train']
        images_train = data['images_train']
        labels_train = data['labels_train']
        
        aud_train = aud_train.detach().float()
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach().long()
        for c in range(channel[1]):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]
        dst_train = CombTensorDataset(aud_train, images_train, labels_train, args)

    else:
        exit('unknown dataset: %s'%dataset)
    
    return channel, im_size, num_classes, mean, std, dst_train

def get_class_train_dataset(dataset, class_num, args):

    if dataset == 'VGG_subset':
        data = torch.load(f'data/classwise_train_data/class_wise_vgg_subset/train_{class_num}.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # Train
        im_train = data['images']
        aud_train = data['audio']
        labels_train = torch.tensor([class_num]*aud_train.shape[0], dtype=torch.long)
        
        aud_train = aud_train.detach().float()
        im_train = im_train.detach().float() / 255.0
        for c in range(channel[1]):
            im_train[:, c] = (im_train[:, c] - mean[c]) / std[c]
        dst_train = CombTensorDataset(aud_train, im_train, labels_train, args)

    elif dataset == 'VGG':
        data = torch.load(f'data/classwise_train_data/class_wise_vgg/train_{class_num}.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 10

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # Train
        im_train = data['images']
        aud_train = data['audio']
        labels_train = torch.tensor([class_num]*aud_train.shape[0], dtype=torch.long)
        
        aud_train = aud_train.detach().float()
        im_train = im_train.detach().float() / 255.0
        for c in range(channel[1]):
            im_train[:, c] = (im_train[:, c] - mean[c]) / std[c]
        dst_train = CombTensorDataset(aud_train, im_train, labels_train, args)        

    elif dataset == 'AVE':
        data = torch.load(f'data/classwise_train_data/class_wise_ave/train_{class_num}.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # Train
        im_train = data['images']
        aud_train = data['audio']
        labels_train = torch.tensor([class_num]*aud_train.shape[0], dtype=torch.long)
        
        aud_train = aud_train.detach().float()
        im_train = im_train.detach().float() / 255.0
        for c in range(channel[1]):
            im_train[:, c] = (im_train[:, c] - mean[c]) / std[c]
        dst_train = CombTensorDataset(aud_train, im_train, labels_train, args)

    elif dataset == 'Music_21':
        data = torch.load(f'data/classwise_train_data/class_wise_music_21/train_{class_num}.pt', map_location='cpu')
        mean = [0.425, 0.396, 0.370]
        std =  [0.229, 0.224, 0.221]
        num_classes = 28

        aud_channel = 1
        im_channel = 3
        channel = [aud_channel, im_channel]

        aud_size = (128, 56)
        im_size = (224, 224)
        im_size = [aud_size, im_size]
        
        # Train
        im_train = data['images']
        aud_train = data['audio']
        labels_train = torch.tensor([class_num]*aud_train.shape[0], dtype=torch.long)
        
        aud_train = aud_train.detach().float()
        im_train = im_train.detach().float() / 255.0
        for c in range(channel[1]):
            im_train[:, c] = (im_train[:, c] - mean[c]) / std[c]
        dst_train = CombTensorDataset(aud_train, im_train, labels_train, args)

    else:
        exit('unknown dataset: %s'%dataset)

    return channel, im_size, dst_train, mean, std, num_classes

def get_herd_path(dataset):
    if dataset == 'VGG_subset':
        return 'data/herding_data/VGG_subset_herd_idx_dict.pkl'
    elif dataset == 'VGG':
        return 'data/herding_data/VGG_herd_idx_dict.cleaned.pkl'
    elif dataset == 'AVE':
        return 'data/herding_data/AVE_herd_idx_dict.pkl'
    elif dataset == 'Music_21_center' or dataset == 'Music_21':
        return 'data/herding_data/Music_21_center_herd_idx_dict.pkl'
    else:
        exit('unknown dataset: %s'%dataset)

def get_herd_path_classwise(dataset):
    if dataset == 'VGG_subset':
        return 'data/herding_data/VGG_subset_herd_idx_dict_local_exp_0.pkl'
    elif dataset == 'VGG':
        return 'data/herding_data/VGG_herd_idx_dict_local_exp_0.pkl'
    elif dataset == 'AVE':
        return 'data/herding_data/AVE_herd_idx_dict_exp_0.pkl'
    elif dataset == 'Music_21_center' or dataset == 'Music_21':
        return 'data/herding_data/Music_21_center_herd_idx_dict_local_exp_0.pkl'
    else:
        exit('unknown dataset: %s'%dataset)
class CombTensorDataset(Dataset):
    def __init__(self, audio, images, labels, args):
        self.args = args
        self.labels = labels.detach()
        self.audio = None
        self.images = None

        if args.input_modality in ['a', 'av']:
            self.audio = audio.detach().float()
            # 保证音频 shape = [*, 1, 128, 204]
            if self.audio.shape[-2] == 128 and self.audio.shape[-1] != 204:
                print(f"[CombTensorDataset] audio origin shape: {self.audio.shape}")
                self.audio = pad_mel_spec(self.audio, 204)
                print(f"[CombTensorDataset] audio padded shape: {self.audio.shape}")

        if args.input_modality in ['v', 'av']:
            self.images = images.detach().float()

    def __getitem__(self, index):
        audio, frame = torch.zeros(1), torch.zeros(1)
        label = self.labels[index]
        if self.args.input_modality in ['a', 'av']:
            audio = self.audio[index]
        if self.args.input_modality in ['v', 'av']:
            frame = self.images[index]
        ret_dict = {'frame': frame, 'audio': audio, 'label': label}
        return ret_dict

    def __len__(self):
        return self.labels.shape[0]


def number_sign_augment(image_syn):
    half_h, half_w = image_syn.shape[2]//2, image_syn.shape[3]//2
    a, b, c, d = image_syn[:, :, :half_h, :half_w].clone(), image_syn[:, :, half_h:, :half_w].clone(), image_syn[:, :, :half_h, half_w:].clone(), image_syn[:, :, half_h:, half_w:].clone()
    a, b, c, d = F.upsample(a, scale_factor=2, mode='bilinear'), F.upsample(b, scale_factor=2, mode='bilinear'), \
        F.upsample(c, scale_factor=2, mode='bilinear'), F.upsample(d, scale_factor=2, mode='bilinear')
    image_syn_augmented = torch.concat([a, b, c, d], dim=0)
    return image_syn_augmented

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

class VGGStreamDataset(Dataset):
    """流式VGG数据集，支持内存优化的数据加载"""
    def __init__(self, data_path, args, max_samples=None, preload_fraction=0.1):
        self.args = args
        self.data_path = data_path
        self.max_samples = max_samples
        self.preload_fraction = preload_fraction
        
        # 加载数据元信息
        print(f"正在加载VGG数据元信息: {data_path}")
        with torch.no_grad():
            data_info = torch.load(data_path, map_location='cpu')
            
            # 限制样本数量（如果指定）
            total_samples = len(data_info['labels_train'])
            if max_samples and max_samples < total_samples:
                self.total_samples = max_samples
                print(f"限制样本数量: {max_samples}/{total_samples}")
            else:
                self.total_samples = total_samples
                print(f"使用全部样本: {total_samples}")
            
            # 存储标签（相对较小，可以全部加载）
            self.labels = data_info['labels_train'][:self.total_samples].detach().long()
            
            # 存储音频和图像数据的引用，但不立即加载
            self.audio_data_ref = data_info['audio_train'][:self.total_samples]
            self.images_data_ref = data_info['images_train'][:self.total_samples]
            
            # 预加载一小部分数据用于快速访问
            preload_size = max(1, int(self.total_samples * preload_fraction))
            self.preload_size = min(preload_size, 1000)  # 最多预加载1000个
            
            print(f"预加载前{self.preload_size}个样本到内存...")
            self.audio_preload = self.audio_data_ref[:self.preload_size].detach().float()
            self.images_preload = self.images_data_ref[:self.preload_size].detach().float() / 255.0
            
            # 数据预处理参数
            self.mean = [0.425, 0.396, 0.370]
            self.std = [0.229, 0.224, 0.221]
            
            # 预处理预加载的数据
            self._preprocess_data()
            
    def _preprocess_data(self):
        """预处理预加载的数据"""
        # 音频预处理
        if self.audio_preload.shape[-2] == 128 and self.audio_preload.shape[-1] != 204:
            self.audio_preload = pad_mel_spec(self.audio_preload, 204)
        
        # 图像预处理
        if self.images_preload.shape[-2:] != (224, 224):
            self.images_preload = F.interpolate(
                self.images_preload, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # 图像标准化
        for c in range(3):
            self.images_preload[:, c] = (self.images_preload[:, c] - self.mean[c]) / self.std[c]
    
    def _load_sample(self, index):
        """按需加载单个样本"""
        if index < self.preload_size:
            # 使用预加载的数据
            audio = self.audio_preload[index]
            image = self.images_preload[index]
        else:
            # 动态加载数据
            audio = self.audio_data_ref[index].detach().float()
            image = self.images_data_ref[index].detach().float() / 255.0
            
            # 确保图像有正确的维度 [C, H, W]
            if len(image.shape) == 3:
                # 已经是 [C, H, W] 格式
                pass
            elif len(image.shape) == 4:
                # [1, C, H, W] 格式，squeeze掉batch维度
                image = image.squeeze(0)
            
            # 音频预处理
            if audio.shape[-2] == 128 and audio.shape[-1] != 204:
                audio = pad_mel_spec(audio, 204)
            
            # 图像预处理
            if image.shape[-2:] != (224, 224):
                image = F.interpolate(
                    image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                ).squeeze(0)
            
            # 确保图像是3通道
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] != 3:
                # 如果有其他通道数，只取前3个通道
                image = image[:3]
            
            # 图像标准化
            for c in range(3):
                image[c] = (image[c] - self.mean[c]) / self.std[c]
        
        return audio, image

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            # Handle batch indexing
            batch_size = len(index)
            audio_batch = []
            image_batch = []
            label_batch = []
            
            for idx in index:
                if self.args.input_modality == 'a':
                    audio, _ = self._load_sample(idx)
                    image = torch.zeros(3, 224, 224)
                elif self.args.input_modality == 'v':
                    _, image = self._load_sample(idx)
                    audio = torch.zeros(1, 128, 204)
                else:  # 'av'
                    audio, image = self._load_sample(idx)
                
                label = self.labels[idx]
                audio_batch.append(audio)
                image_batch.append(image)
                label_batch.append(label)
            
            # Stack tensors to create batches
            audio_batch = torch.stack(audio_batch)
            image_batch = torch.stack(image_batch)
            label_batch = torch.tensor(label_batch)
            
            return {'frame': image_batch, 'audio': audio_batch, 'label': label_batch}
        else:
            # Handle single indexing
            if self.args.input_modality == 'a':
                audio, _ = self._load_sample(index)
                image = torch.zeros(3, 224, 224)
            elif self.args.input_modality == 'v':
                _, image = self._load_sample(index)
                audio = torch.zeros(1, 128, 204)
            else:  # 'av'
                audio, image = self._load_sample(index)
            
            label = self.labels[index]
            return {'frame': image, 'audio': audio, 'label': label}

    def __len__(self):
        return self.total_samples
