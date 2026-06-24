# -*- coding: utf-8 -*-
# Author: 
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.fuse_modules.lamma import LAMMA, LAMMA2, LAMMA3
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.tools.sd_lamma import SupplyDemandLAMMAComm
from opencood.models.fuse_modules.fusion_in_one import (
    MaxFusion, AttFusion, DiscoFusion, 
    V2VNetFusion, V2XViTFusion, 
    CoBEVT, Where2commFusion, Who2comFusion
)
from opencood.tools import train_utils
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import time
import torchvision

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum
from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, BevEncodeSSFusion, Up, CamEncode, BevEncode
from matplotlib import pyplot as plt

try:
    from opencood.tools.light_sad import LightSADDispatcher, action_to_runtime_mask
except Exception:
    LightSADDispatcher = None
    action_to_runtime_mask = None


class PointPillarLSSLamma2PyramidFusion(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarLSSLamma2PyramidFusion, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.light_sad_enabled = False
        self.light_sad = None
        light_sad_cfg = args.get("light_sad", None)
        if light_sad_cfg and light_sad_cfg.get("enabled", False):
            if LightSADDispatcher is None or action_to_runtime_mask is None:
                raise ImportError("LightSADDispatcher import failed.")
            self.light_sad_enabled = True
            self.light_sad = LightSADDispatcher(light_sad_cfg)
            print("[Light-SAD] enabled:", light_sad_cfg)

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls
                
            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            # """
            # Shrink conv building
            # """
            # setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))
            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }
            

            # freeze the pretrained modules
            setattr(self, f"encoder_{modality_name}_freeze", model_setting['encoder_args'].get('freeze', False))
            if eval(f"self.encoder_{modality_name}_freeze"):
                print(f"Freeze {modality_name} encoder")
                for param in getattr(self, f"encoder_{modality_name}").parameters():
                    param.requires_grad = False
            setattr(self, f"backbone_{modality_name}_freeze", model_setting['backbone_args'].get('freeze', False))
            if eval(f"self.backbone_{modality_name}_freeze"):
                print(f"Freeze {modality_name} backbone")
                for param in getattr(self, f"backbone_{modality_name}").parameters():
                    param.requires_grad = False
            # setattr(self, f"shrinker_{modality_name}_freeze", model_setting['shrink_header'].get('freeze', False))
            # if eval(f"self.shrinker_{modality_name}_freeze"):
            #     print(f"Freeze {modality_name} shrinker")
            #     for param in getattr(self, f"shrinker_{modality_name}").parameters():
            #         param.requires_grad = False
            setattr(self, f"aligner_{modality_name}_freeze", model_setting['aligner_args'].get('freeze', False))
            if eval(f"self.aligner_{modality_name}_freeze"):
                print(f"Freeze {modality_name} aligner")
                for param in getattr(self, f"aligner_{modality_name}").parameters():
                    param.requires_grad = False


        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        # self.supervise_single = False
        # if args.get("supervise_single", False):
        #     self.supervise_single = True
        #     in_head_single = args['in_head_single']
        #     setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
        #     setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
        #     setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))

        """
        multi-modal fusion
        """
        try:
            self.mm_pool_method = args['mm_pooling']['pool_method']
        except:
            self.mm_pool_method = None
        if self.mm_pool_method == "max":
            self.mm_pooling = nn.Sequential(
                nn.MaxPool2d(kernel_size=args['mm_pooling']['pool_kernel_size']),
                nn.ReLU()
            )
        elif self.mm_pool_method == "avg":
            self.mm_pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=args['mm_pooling']['pool_kernel_size']),
                nn.ReLU()
            )

        self.voxel_size = args['voxel_size']
        self.fH = round(self.H / self.voxel_size[0] / args['lamma']['feature_stride'])
        self.fW = round(self.W / self.voxel_size[1] / args['lamma']['feature_stride'])
        # if self.mm_pool_method:
        #     self.fH = round(self.fH / args['mm_pooling']['pool_kernel_size'])
        #     self.fW = round(self.fW / args['mm_pooling']['pool_kernel_size'])
        if args['mm_fusion_method'] == "lamma":
            self.mm_fusion = LAMMA(args['lamma'], H=round(self.fH), W=round(self.fW))
        elif args['mm_fusion_method'] == "lamma2":
            self.mm_fusion = LAMMA2(args['lamma'], H=round(self.fH), W=round(self.fW))
        elif args['mm_fusion_method'] == "lamma3":
            self.mm_fusion = LAMMA3(args['lamma'], H=round(self.fH), W=round(self.fW))

        # freeze multi-modal fusion
        setattr(self, f"mm_fusion_freeze", args['lamma'].get('freeze', False))
        if self.mm_fusion_freeze:
            print("Freeze multi-modal fusion")
            for param in self.mm_fusion.parameters():
                param.requires_grad = False

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        self.sd_lamma_enabled = False
        self.sd_lamma_comm = None
        sd_lamma_cfg = args.get("sd_lamma", None)
        if sd_lamma_cfg:
            sd_lamma_cfg = dict(sd_lamma_cfg)
            sd_lamma_cfg.setdefault("lidar_range", self.cav_range)
            sd_lamma_cfg.setdefault("voxel_size", self.voxel_size)
            self.sd_lamma_enabled = bool(sd_lamma_cfg.get("enabled", False))
            self.sd_lamma_comm = SupplyDemandLAMMAComm(sd_lamma_cfg)
            if self.sd_lamma_enabled:
                print("[SD-LAMMA] enabled:", sd_lamma_cfg)
        
        # freeze multi-modal fusion
        setattr(self, f"ma_fusion_freeze", args['fusion_backbone'].get('freeze', False))
        if self.ma_fusion_freeze:
            print("Freeze pyramid fusion")
            for param in self.pyramid_backbone.parameters():
                param.requires_grad = False


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            setattr(self, f"shrink_freeze", args['shrink_header'].get('freeze', False))
            if self.shrink_freeze:
                print(f"Freeze shrink conv")
                for param in self.shrink_conv.parameters():
                    param.requires_grad = False

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        if args.get("head_freeze", False):
            self.head_freeze = True
            print(f"Freeze task heads")
            for param in getattr(self, f"cls_head").parameters():
                param.requires_grad = False
            for param in getattr(self, f"reg_head").parameters():
                param.requires_grad = False
            for param in getattr(self, f"dir_head").parameters():
                param.requires_grad = False
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])
            self.model_train_init()

        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        # split_x = torch.split(x, cum_sum_len[:-1])
        # TypeError: split_with_sizes(): argument 'split_sizes' (position 2) must be tuple of ints, not Tensor
        return split_x

    @staticmethod
    def _record_len_total(record_len):
        if torch.is_tensor(record_len):
            return int(record_len.detach().cpu().view(-1).sum().item())
        if isinstance(record_len, (list, tuple)):
            return int(sum(int(x) for x in record_len))
        return int(record_len)

    @staticmethod
    def _expand_actions_for_cavs(actions, total_cavs):
        if isinstance(actions, str):
            parts = [x.strip() for x in actions.split(',') if x.strip()]
        else:
            parts = [str(x) for x in (actions or [])]
        if not parts:
            parts = ["LC"]
        return [parts[idx % len(parts)] for idx in range(total_cavs)]

    def _build_runtime_agent_modality_list(self, data_dict, record_len, active_actions=None):
        total_cavs = self._record_len_total(record_len)
        if active_actions is not None:
            actions = self._expand_actions_for_cavs(active_actions, total_cavs)
            modality_list = []
            for action in actions:
                action = str(action).upper()
                if action == "C":
                    modality_list.append("m2")
                elif action == "L":
                    modality_list.append("m1")
                else:
                    modality_list.append("m1_m2")
            return modality_list

        existing = data_dict.get('agent_modality_list', None)
        if isinstance(existing, list) and len(existing) == total_cavs:
            return existing
        return ["m1_m2" for _ in range(total_cavs)]

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        runtime_profile = {
            "light_sad_enabled": bool(self.light_sad_enabled),
            "policy_inference_time_ms": 0.0,
            "encoder_time_ms": 0.0,
            "lamma_time_ms": 0.0,
            "sd_lamma_time_ms": 0.0,
            "pyramid_fusion_time_ms": 0.0,
        }
        light_sad_info = None
        light_sad_action = "LC"
        light_sad_actions = None

        if self.light_sad_enabled:
            policy_start = time.perf_counter()
            light_sad_info = self.light_sad.dispatch(data_dict, record_len=data_dict.get("record_len", None))
            runtime_profile["policy_total_time_ms"] = (time.perf_counter() - policy_start) * 1000.0
            policy_time = light_sad_info.get("policy_inference_time_ms", 0.0)
            if isinstance(policy_time, list):
                policy_time = sum(float(x or 0.0) for x in policy_time)
            runtime_profile["policy_inference_time_ms"] = float(policy_time or 0.0)
            light_sad_action = light_sad_info.get("action", "LC")
            light_sad_actions = light_sad_info.get("actions", None)
            if self.light_sad.cfg.log:
                if light_sad_actions is not None:
                    print(f"[Light-SAD] mode={light_sad_info.get('mode')} policy={light_sad_info.get('policy_type')} actions={light_sad_actions} reasons={light_sad_info.get('reasons')}")
                else:
                    print(f"[Light-SAD] policy={light_sad_info.get('policy_type')} action={light_sad_action}, reason={light_sad_info.get('reason')}")
                if self.light_sad.cfg.log_policy_prob and light_sad_info.get("action_probs") is not None:
                    print(f"[Light-SAD] action_probs={light_sad_info.get('action_probs')} fallback={light_sad_info.get('fallback_used')}")

        record_len = data_dict['record_len']
        active_actions = light_sad_actions if light_sad_actions is not None else [light_sad_action]
        agent_modality_list = self._build_runtime_agent_modality_list(
            data_dict,
            record_len,
            active_actions if self.light_sad_enabled else None,
        )
        run_lidar = any("L" in str(action) for action in active_actions)
        run_camera = any("C" in str(action) for action in active_actions)
        runtime_profile["run_lidar_branch"] = bool(run_lidar)
        runtime_profile["run_camera_branch"] = bool(run_camera)
        runtime_profile["selected_action"] = light_sad_actions if light_sad_actions is not None else light_sad_action

        available_modality_dict = {name: 1 for name in self.modality_name_list}
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        # print("record_len:", record_len)
        # print(agent_modality_list)

        for modality_name in self.modality_name_list:
            if modality_name == 'm1':
                if 'processed_lidar' in data_dict:
                    data_dict[f"inputs_{modality_name}"] = data_dict.pop('processed_lidar')
            elif modality_name == 'm2':
                if 'image_inputs' in data_dict:
                    data_dict[f"inputs_{modality_name}"] = data_dict.pop('image_inputs')
            else:
                raise ValueError(f"Modality name {modality_name} not supported.")

        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in available_modality_dict:
                continue
            if modality_name == "m1" and not run_lidar:
                continue
            if modality_name == "m2" and not run_camera:
                continue

            modality_start = time.perf_counter()
            if eval(f"self.encoder_{modality_name}_freeze"):
                eval(f"self.encoder_{modality_name}").eval()
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)                               # m1: torch.Size([4, 64, 256, 256])  m2: torch.Size([4, 128, 256, 256])
            
            if eval(f"self.backbone_{modality_name}_freeze"):
                eval(f"self.backbone_{modality_name}").eval()
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']  # m1: torch.Size([4, 64, 128, 128]) m2: torch.Size([4, 64, 128, 128])
            
            # if eval(f"self.shrinker_{modality_name}_freeze"):
            #     eval(f"self.shrinker_{modality_name}").eval()
            # feature = eval(f"self.shrinker_{modality_name}")(feature)                                              
            
            if eval(f"self.aligner_{modality_name}_freeze"):
                eval(f"self.aligner_{modality_name}").eval()
            feature = eval(f"self.aligner_{modality_name}")(feature)                                                # m1: torch.Size([3, 64, 128, 128]) m2: torch.Size([3, 64, 128, 128])
            
            modality_feature_dict[modality_name] = feature
            elapsed = (time.perf_counter() - modality_start) * 1000.0
            runtime_profile[f"{modality_name}_encoder_time_ms"] = elapsed
            runtime_profile["encoder_time_ms"] += elapsed

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in available_modality_dict and modality_name in modality_feature_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature) # m1:torch.Size([4, 256, 128, 128]) m2:torch.Size([4, 256, 128, 128])
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Fuse multimodalities.
        """
        if self.mm_pool_method == 'max' or self.mm_pool_method == 'avg':
            pc_feature = self.mm_pooling(modality_feature_dict['m1']) if run_lidar else None
            img_fused_feature = self.mm_pooling(modality_feature_dict['m2']) if run_camera else None
        else:
            pc_feature = modality_feature_dict['m1'] if run_lidar else None
            img_fused_feature = modality_feature_dict['m2'] if run_camera else None

        if pc_feature is None and img_fused_feature is None:
            raise RuntimeError("Light-SAD disabled both modalities, not supported in module-1.")
        if pc_feature is None:
            pc_feature = torch.zeros_like(img_fused_feature)
        if img_fused_feature is None:
            img_fused_feature = torch.zeros_like(pc_feature)

        pc_feature = torch.stack(self.regroup(pc_feature, record_len)) # torch.Size([1, 3, 64, 64, 64])
        img_fused_feature = torch.stack(self.regroup(img_fused_feature, record_len)) # torch.Size([1, 3, 64, 64, 64])
        runtime_modality_mask = None
        if self.light_sad_enabled:
            B = pc_feature.shape[0]
            N = pc_feature.shape[1]
            if light_sad_actions is not None:
                runtime_modality_mask = action_to_runtime_mask(
                    light_sad_actions,
                    batch_size=B,
                    cav_num=N,
                    device=pc_feature.device,
                    record_len=record_len,
                )
            else:
                runtime_modality_mask = action_to_runtime_mask(light_sad_action, B, N, pc_feature.device, record_len=record_len)
        # mm_feature_2d, _, _ = self.mm_fusion(pc_feature, img_fused_feature)
        lamma_start = time.perf_counter()
        mm_feature_2d, _, _ = self.mm_fusion(
            img_fused_feature,
            pc_feature,
            runtime_modality_mask=runtime_modality_mask
        ) # torch.Size([3, 64, 64, 64])
        runtime_profile["lamma_time_ms"] = (time.perf_counter() - lamma_start) * 1000.0

        if self.compress:
            mm_feature_2d = self.compressor(mm_feature_2d)

        sd_lamma_debug = None
        if self.sd_lamma_enabled and self.sd_lamma_comm is not None:
            sd_start = time.perf_counter()
            mm_feature_2d, sd_lamma_debug = self.sd_lamma_comm(
                mm_feature_2d,
                record_len,
                affine_matrix,
                data_dict=data_dict,
                light_sad_info=light_sad_info,
                runtime_modality_mask=runtime_modality_mask,
                confidence_head=getattr(self.pyramid_backbone, "single_head_0", None),
            )
            runtime_profile["sd_lamma_time_ms"] = (time.perf_counter() - sd_start) * 1000.0

        """
        Feature Fusion (multiscale).
        """
        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        if self.ma_fusion_freeze:
            self.pyramid_backbone.eval()
        pyramid_start = time.perf_counter()
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                mm_feature_2d,
                                                record_len,
                                                affine_matrix,
                                                agent_modality_list,
                                                self.cam_crop_info
                                            ) # torch.Size([1, 256, 64, 64])
        runtime_profile["pyramid_fusion_time_ms"] = (time.perf_counter() - pyramid_start) * 1000.0

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature) # torch.Size([1, 2, 64, 64])
        reg_preds = self.reg_head(fused_feature) # torch.Size([1, 14, 64, 64])
        dir_preds = self.dir_head(fused_feature) # torch.Size([1, 4, 64, 64])

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds,
                            'pc_feature': pc_feature,
                            'img_feature': img_fused_feature,})
        if light_sad_info is not None:
            output_dict["light_sad_action"] = light_sad_action
            output_dict["light_sad_reason"] = light_sad_info.get("reason", "")
            output_dict["light_sad_actions"] = light_sad_actions
            output_dict["light_sad_reasons"] = light_sad_info.get("reasons", None)
            output_dict["light_sad_mode"] = light_sad_info.get("mode", "batch")
            output_dict["light_sad_state_summary"] = light_sad_info.get("state_summary", {})
            output_dict["light_sad_reliability"] = light_sad_info.get("reliability", None)
            output_dict["light_sad_reliabilities"] = light_sad_info.get("reliabilities", None)
            output_dict["light_sad_policy_type"] = light_sad_info.get("policy_type", None)
            output_dict["light_sad_action_probs"] = light_sad_info.get("action_probs", None)
            output_dict["light_sad_action_logits"] = light_sad_info.get("action_logits", None)
            output_dict["light_sad_feature_names"] = light_sad_info.get("feature_names", None)
            output_dict["light_sad_feature_vector"] = light_sad_info.get("feature_vector", None)
            output_dict["light_sad_fallback_used"] = light_sad_info.get("fallback_used", None)
            output_dict["light_sad_fallback_reason"] = light_sad_info.get("fallback_reason", light_sad_info.get("fallback_reasons", None))
        if sd_lamma_debug is not None:
            output_dict["sd_lamma_debug"] = sd_lamma_debug
        output_dict["runtime_profile"] = runtime_profile
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})
        return output_dict

