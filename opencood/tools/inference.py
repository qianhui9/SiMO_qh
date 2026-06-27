# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, simple_vis
from opencood.utils.common_utils import update_dict
try:
    from opencood.tools.light_sad import HistoryConfidenceBuffer
except Exception:
    HistoryConfidenceBuffer = None
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default=None,
                        help="Optional override for detection range, e.g. 51.2,51.2. If omitted, use the checkpoint/config range.")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument("--light_sad_enable", action="store_true",    # 启用 Light-SAD 调度器
                        help="Enable Light-SAD runtime modality scheduler.")
    parser.add_argument("--light_sad_force_action", default=None,
                        choices=["L", "C", "LC"],
                        help="Force Light-SAD action for smoke test.")
    parser.add_argument("--light_sad_log", action="store_true",   # 打印每帧调度动作和原因
                        help="Print Light-SAD action and reason.")
    parser.add_argument("--light_sad_per_cav", action="store_true",   # 启用 per-CAV 调度，也就是为每个协作车辆分别输出 L / C / LC
                        help="Enable per-CAV Light-SAD scheduling.")
    parser.add_argument("--light_sad_use_history", action="store_true",  # 启用历史检测置信度。当前帧推理结束后，用 pred_score 更新 history buffer，下一帧调度时再使用
                        help="Use previous-frame detection score history for Light-SAD.")
    parser.add_argument("--light_sad_use_local_reliability", action="store_true",  # 启用局部可靠性代理，根据 LiDAR voxel 分布和 Camera 质量构造粗粒度 BEV reliability map
                        help="Use coarse local reliability summaries for Light-SAD.")
    parser.add_argument("--light_sad_policy", default="emc2_rule",
                        choices=["force", "emc2_rule", "emc2_rule_history", "emc2_rule_local", "emc2_rule_full", "learned_mlp", "hybrid"],
                        help="Light-SAD decision policy.")
    parser.add_argument("--light_sad_learned_ckpt", default=None,
                        help="Path to learned Light-SAD policy checkpoint.")
    parser.add_argument("--light_sad_feature_norm_path", default=None,
                        help="Optional feature normalization JSON for learned policy.")
    parser.add_argument("--light_sad_temperature", type=float, default=None,
                        help="Softmax temperature for learned policy inference.")
    parser.add_argument("--light_sad_safe_fallback", action="store_true", default=None,
                        help="Enable learned policy safety fallback.")
    parser.add_argument("--light_sad_disable_safe_fallback", action="store_false", dest="light_sad_safe_fallback",
                        help="Disable learned policy safety fallback and raise on missing/invalid checkpoint.")
    parser.add_argument("--light_sad_min_conf_margin", type=float, default=None,
                        help="Minimum top-1/top-2 probability margin used by hybrid fallback.")
    parser.add_argument("--light_sad_log_policy_prob", action="store_true",
                        help="Print learned Light-SAD action probabilities.")
    parser.add_argument("--light_sad_log_feature_vector", action="store_true",
                        help="Attach raw learned-policy feature vectors to debug dumps.")
    parser.add_argument("--light_sad_force_actions", default=None,
                        help="Comma-separated per-CAV force actions, e.g. L,LC,C. Values cycle if fewer than CAVs.")
    parser.add_argument("--light_sad_dump_state", action="store_true",
                        help="Dump Light-SAD state/debug summaries as JSONL.")
    parser.add_argument("--light_sad_dump_path", default=None,
                        help="Path for --light_sad_dump_state JSONL output.")
    parser.add_argument("--light_sad_max_batches", type=int, default=None,   # 只跑前 5 个 batch，做 smoke test
                        help="Optional smoke-test limit.")
    parser.add_argument("--sd_lamma_enable", action="store_true",
                        help="Enable SD-LAMMA supply-demand communication masks.")
    parser.add_argument("--sd_lamma_log", action="store_true",
                        help="Print SD-LAMMA demand/supply/communication summaries.")
    parser.add_argument("--sd_lamma_budget_mode", default=None,
                        choices=["threshold", "topk"],
                        help="Override SD-LAMMA network budget mode.")
    parser.add_argument("--sd_lamma_max_comm_ratio", type=float, default=None,
                        help="Optional max selected communication ratio across collaborator BEV cells.")
    parser.add_argument("--sd_lamma_demand_topk_ratio", type=float, default=None,
                        help="Optional top-k ratio for ego demand mask generation.")
    parser.add_argument("--sd_lamma_supply_threshold", type=float, default=None,
                        help="Optional threshold for collaborator supply confidence.")
    parser.add_argument("--sd_lamma_no_redundancy", action="store_true",
                        help="Disable CodeFilling-style remaining-demand de-duplication.")
    parser.add_argument("--sd_lamma_allow_overlap", action="store_true",
                        help="Allow multiple collaborators to fill the same ego BEV region.")
    parser.add_argument("--sd_lamma_save_masks", action="store_true",
                        help="Attach SD-LAMMA demand/supply/selection masks to model debug output.")
    parser.add_argument("--sd_lamma_mode", default=None,
                        choices=["pairwise", "broadcast"],
                        help="Select SD-LAMMA communication mode. Defaults to yaml or pairwise.")
    parser.add_argument("--sd_lamma_broadcast_enable", action="store_true",
                        help="Enable BROAD-SD-LAMMA and set sd_lamma.mode=broadcast.")
    parser.add_argument("--sd_lamma_broadcast_method", default=None,
                        choices=["soft_or", "vra"],
                        help="Broadcast demand estimator.")
    parser.add_argument("--sd_lamma_num_virtual_receivers", type=int, default=None,
                        help="Number of virtual receiver tokens for BROAD-SD-LAMMA.")
    parser.add_argument("--sd_lamma_receiver_gating", action="store_true",
                        help="Enable ego-side local gating after receiving broadcast messages.")
    parser.add_argument("--sd_lamma_save_broadcast_debug", action="store_true",
                        help="Save broadcast demand/mask/utility debug tensors.")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes and opt.range is not None:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    if opt.light_sad_enable:
        model_args = hypes["model"]["args"]
        model_args["light_sad"] = model_args.get("light_sad", {})
        if opt.light_sad_force_action and opt.light_sad_force_actions:
            print("[Light-SAD] both force_action and force_actions provided; per-CAV force_actions takes priority.")
            opt.light_sad_force_action = None
        model_args["light_sad"]["enabled"] = True
        model_args["light_sad"]["policy"] = "force" if (opt.light_sad_force_action or opt.light_sad_force_actions) else opt.light_sad_policy
        model_args["light_sad"]["force_action"] = opt.light_sad_force_action
        model_args["light_sad"]["force_actions"] = opt.light_sad_force_actions
        model_args["light_sad"]["log"] = opt.light_sad_log
        model_args["light_sad"]["learned_ckpt"] = opt.light_sad_learned_ckpt
        model_args["light_sad"]["feature_norm_path"] = opt.light_sad_feature_norm_path
        if opt.light_sad_temperature is not None:
            model_args["light_sad"]["temperature"] = opt.light_sad_temperature
        if opt.light_sad_safe_fallback is not None:
            model_args["light_sad"]["safe_fallback"] = opt.light_sad_safe_fallback
        if opt.light_sad_min_conf_margin is not None:
            model_args["light_sad"]["min_conf_margin"] = opt.light_sad_min_conf_margin
        model_args["light_sad"]["log_policy_prob"] = opt.light_sad_log_policy_prob
        model_args["light_sad"]["log_feature_vector"] = opt.light_sad_log_feature_vector
        if opt.light_sad_per_cav or opt.light_sad_force_actions:
            model_args["light_sad"]["per_cav"] = True
        model_args["light_sad"]["use_history"] = opt.light_sad_use_history
        model_args["light_sad"]["use_local_reliability"] = opt.light_sad_use_local_reliability
        model_args["light_sad"]["debug_dump_state"] = opt.light_sad_dump_state
        model_args["light_sad"]["debug_dump_path"] = opt.light_sad_dump_path

    sd_lamma_cli_requested = any([
        opt.sd_lamma_enable,
        opt.sd_lamma_log,
        opt.sd_lamma_budget_mode is not None,
        opt.sd_lamma_max_comm_ratio is not None,
        opt.sd_lamma_demand_topk_ratio is not None,
        opt.sd_lamma_supply_threshold is not None,
        opt.sd_lamma_no_redundancy,
        opt.sd_lamma_allow_overlap,
        opt.sd_lamma_save_masks,
        opt.sd_lamma_mode is not None,
        opt.sd_lamma_broadcast_enable,
        opt.sd_lamma_broadcast_method is not None,
        opt.sd_lamma_num_virtual_receivers is not None,
        opt.sd_lamma_receiver_gating,
        opt.sd_lamma_save_broadcast_debug,
    ])
    if sd_lamma_cli_requested:
        model_args = hypes["model"]["args"]
        sd_cfg = model_args.get("sd_lamma", {})
        model_args["sd_lamma"] = sd_cfg
        if opt.sd_lamma_enable or opt.sd_lamma_broadcast_enable:
            sd_cfg["enabled"] = True
        sd_cfg.setdefault("mode", "pairwise")
        sd_cfg.setdefault("network", {})
        sd_cfg.setdefault("demand", {})
        sd_cfg.setdefault("supply", {})
        sd_cfg.setdefault("redundancy", {})
        sd_cfg.setdefault("debug", {})
        sd_cfg.setdefault("mask", {})
        sd_cfg.setdefault("broadcast", {})
        sd_cfg["broadcast"].setdefault("receiver_gating", {})
        sd_cfg["broadcast"].setdefault("debug", {})
        if opt.sd_lamma_broadcast_enable:
            sd_cfg["mode"] = "broadcast"
            sd_cfg["broadcast"]["enabled"] = True
        if opt.sd_lamma_mode is not None:
            sd_cfg["mode"] = opt.sd_lamma_mode
        if opt.sd_lamma_budget_mode is not None:
            sd_cfg["network"]["budget_mode"] = opt.sd_lamma_budget_mode
        if opt.sd_lamma_max_comm_ratio is not None:
            sd_cfg["network"]["max_comm_ratio"] = opt.sd_lamma_max_comm_ratio
        if opt.sd_lamma_demand_topk_ratio is not None:
            sd_cfg["demand"]["topk_ratio"] = opt.sd_lamma_demand_topk_ratio
        if opt.sd_lamma_supply_threshold is not None:
            sd_cfg["supply"]["confidence_threshold"] = opt.sd_lamma_supply_threshold
        if opt.sd_lamma_no_redundancy:
            sd_cfg["redundancy"]["enabled"] = False
        if opt.sd_lamma_allow_overlap:
            sd_cfg["redundancy"]["allow_overlap"] = True
        if opt.sd_lamma_broadcast_method is not None:
            sd_cfg["broadcast"]["method"] = opt.sd_lamma_broadcast_method
            sd_cfg["broadcast"]["use_vra"] = opt.sd_lamma_broadcast_method == "vra"
        if opt.sd_lamma_num_virtual_receivers is not None:
            sd_cfg["broadcast"]["num_virtual_receivers"] = opt.sd_lamma_num_virtual_receivers
        if opt.sd_lamma_receiver_gating:
            sd_cfg["broadcast"]["receiver_gating"]["enabled"] = True
        if opt.sd_lamma_save_broadcast_debug:
            sd_cfg["debug"]["save_masks"] = True
            sd_cfg["broadcast"]["debug"]["save_broadcast_demand"] = True
            sd_cfg["broadcast"]["debug"]["save_broadcast_mask"] = True
            sd_cfg["broadcast"]["debug"]["save_broadcast_utility"] = True
        sd_cfg["debug"]["log"] = opt.sd_lamma_log
        sd_cfg["debug"]["save_masks"] = opt.sd_lamma_save_masks or sd_cfg["debug"].get("save_masks", False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=0,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    history_buffer = None
    history_missing_score_warned = False
    if opt.light_sad_enable and opt.light_sad_use_history:
        if HistoryConfidenceBuffer is None:
            print("[Light-SAD] HistoryConfidenceBuffer unavailable; history is disabled.")
        else:
            light_sad_cfg = hypes["model"]["args"].get("light_sad", {})
            history_buffer = HistoryConfidenceBuffer(
                topk=light_sad_cfg.get("history_topk", 20),
                stale_limit=light_sad_cfg.get("history_stale_limit", 3),
            )

    
    infer_info = opt.fusion_method + opt.note
    try:
        # if hypes['model']['args']['lamma']['single_mode']:
            # single_mode = hypes['model']['args']['lamma']['single_mode']
        if hypes['model']['args']['single_modality']:
            single_mode = hypes['model']['args']['single_modality']
            infer_info = infer_info + '_' + single_mode
            print(f"Inference with {single_mode}")
    except:
        pass


    for i, batch_data in enumerate(data_loader):
        if opt.light_sad_max_batches is not None and i >= opt.light_sad_max_batches:
            break
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if history_buffer is not None and 'ego' in batch_data:
                batch_data['ego']['light_sad_history'] = history_buffer.get_state()

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            if history_buffer is not None:
                if pred_score is None and not history_missing_score_warned:
                    print("[Light-SAD] pred_score unavailable; history update will be stale until scores appear.")
                    history_missing_score_warned = True
                history_buffer.update(pred_score)
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)

if __name__ == '__main__':
    main()
