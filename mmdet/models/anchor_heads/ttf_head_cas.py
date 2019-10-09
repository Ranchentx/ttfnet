import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
import numpy as np

from mmdet.ops import ModulatedDeformConvPack
from mmdet.core import multi_apply, bbox_areas, force_fp32
from mmdet.core.anchor.guided_anchor_target import calc_region
from mmdet.models.losses import ct_focal_loss, giou_loss
from mmdet.models.utils import (build_norm_layer, bias_init_with_prob, ConvModule)
from mmdet.ops.nms import simple_nms
from .anchor_head import AnchorHead
from ..registry import HEADS

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

@HEADS.register_module
class TTFHead_cas(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 base_down_ratio=32,
                 head_conv=256,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 shortcut_kernel=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 shortcut_cfg=(1, 2, 3),
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 alpha=0.54,
                 alpha_2=0.54,
                 alpha_3=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 hm_weight_2=1.,
                 hm_weight_3=1.,
                 wh_weight=5.,
                 wh_weight_2=5.,
                 wh_weight_3=5.,
                 max_objs=128,
                 upsample_sc=True,

                 ):
        super(AnchorHead, self).__init__()
        assert len(planes) in [2, 3, 4]
        shortcut_num = min(len(inplanes) - 1, len(planes))
        assert shortcut_num == len(shortcut_cfg)
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.planes = planes
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.beta = beta
        self.hm_weight = hm_weight
        self.hm_weight_2 = hm_weight_2
        self.hm_weight_3 = hm_weight_3
        self.wh_weight = wh_weight
        self.wh_weight_2 = wh_weight_2
        self.wh_weight_3 = wh_weight_3
        self.max_objs = max_objs
        self.fp16_enabled = False
        self.upsample_sc = upsample_sc
        self.down_ratio = base_down_ratio // 2 ** len(planes)
        self.num_fg = num_classes - 1
        self.wh_planes = 4 if wh_agnostic else 4 * self.num_fg
        self.base_loc = None

        # repeat upsampling n times. 32x to 4x by default.
        self.deconv_layers = nn.ModuleList([
            self.build_upsample(inplanes[-1], planes[0], norm_cfg=norm_cfg),
            self.build_upsample(planes[0], planes[1], norm_cfg=norm_cfg)
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self.build_upsample(planes[i - 1], planes[i], norm_cfg=norm_cfg))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self.build_shortcut(
            inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
            kernel_size=shortcut_kernel, padding=padding)

        # self.cas_conv2 = BasicConv(self.planes[-1],self.planes[-1], kernel_size=3, padding=1)
        self.cas_conv2 = ModulatedDeformConvPack(self.planes[-1],self.planes[-1], 3, stride=1,
                                       padding=1, dilation=1, deformable_groups=1)
        # self.cas_conv3 = BasicConv(self.planes[-1],self.planes[-1], kernel_size=3, padding=1)
        self.cas_conv3 = ModulatedDeformConvPack(self.planes[-1], self.planes[-1], 3, stride=1,
                                                 padding=1, dilation=1, deformable_groups=1)
        # heads
        self.wh = self.build_head(self.wh_planes, wh_head_conv_num, wh_conv)
        self.hm = self.build_head(self.num_fg, hm_head_conv_num)

        self.wh_2 = self.build_head(self.wh_planes, wh_head_conv_num, wh_conv)
        self.hm_2 = self.build_head(self.num_fg, hm_head_conv_num)

        self.wh_3 = self.build_head(self.wh_planes, wh_head_conv_num, wh_conv)
        self.hm_3 = self.build_head(self.num_fg, hm_head_conv_num)

    def build_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def build_upsample(self, inplanes, planes, norm_cfg=None):
        mdcn = ModulatedDeformConvPack(inplanes, planes, 3, stride=1,
                                       padding=1, dilation=1, deformable_groups=1)

        up = nn.UpsamplingBilinear2d(scale_factor=2)

        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        if self.upsample_sc:
            layers.append(up)

        return nn.Sequential(*layers)

    def build_head(self, out_channel, conv_num=1, head_conv_plane=None):
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        for i in range(conv_num):
            inp = self.planes[-1] if i == 0 else head_conv_plane
            head_convs.append(ConvModule(inp, head_conv_plane, 3, padding=1))

        inp = self.planes[-1] if conv_num <= 0 else head_conv_plane
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    def init_weights(self):
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]
        for i, upsample_layer in enumerate(self.deconv_layers):
            x = upsample_layer(x)
            if i < len(self.shortcut_layers):
                shortcut = self.shortcut_layers[i](feats[-i - 2])
                x = x + shortcut

        hm = self.hm(x)
        wh = F.relu(self.wh(x)) * self.wh_offset_base

        x_2 =self.cas_conv2(x)
        hm_2 = self.hm_2(x_2)
        wh_2 = F.relu(self.wh_2(x_2)) * self.wh_offset_base

        x_3 = self.cas_conv3(x_2)
        hm_3 = self.hm_3(x_3)
        wh_3 = F.relu(self.wh_3(x_3)) * self.wh_offset_base

        return hm, wh, hm_2, wh_2, hm_3, wh_3

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh', 'pred_heatmap_2', 'pred_wh_2','pred_heatmap_3', 'pred_wh_3'))
    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   pred_heatmap_2,
                   pred_wh_2,
                   pred_heatmap_3,
                   pred_wh_3,
                   img_metas,
                   cfg,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        # pred_heatmap_1 = pred_heatmap.detach().sigmoid_()
        pred_heatmap = pred_heatmap_3.detach().sigmoid_()
        # pred_heatmap_3 = pred_heatmap_3.detach().sigmoid_()
        # pred_heatmap = pred_heatmap_1 *pred_heatmap_2 *pred_heatmap_3
        wh = pred_wh_3.detach()

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score

        topk = getattr(cfg, 'max_per_img', 100)
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_fg, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for idx in range(bboxes.shape[0]):
            scores_per_img = scores[idx]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[idx][scores_keep]
            labels_per_img = clses[idx][scores_keep]

            if rescale:
                scale_factor = img_metas[idx]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)
            result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh', 'pred_heatmap_2', 'pred_wh_2','pred_heatmap_3', 'pred_wh_3'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             pred_heatmap_2,
             pred_wh_2,
             pred_heatmap_3,
             pred_wh_3,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss,hm_loss_2, wh_loss_2,hm_loss_3, wh_loss_3 = self.loss_calc(pred_heatmap, pred_wh,pred_heatmap_2,pred_wh_2, pred_heatmap_3,\
             pred_wh_3, *all_targets)
        return {'losses/ttfnet_loss_heatmap': hm_loss, 'losses/ttfnet_loss_wh': wh_loss, \
                'losses/ttfnet_loss_heatmap_2': hm_loss_2, 'losses/ttfnet_loss_wh_2': wh_loss_2, \
                'losses/ttfnet_loss_heatmap_3': hm_loss_3, 'losses/ttfnet_loss_wh_3': wh_loss_3}

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        heatmap_2 = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        heatmap_3 = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        fake_heatmap_2 = gt_boxes.new_zeros((output_h, output_w))
        fake_heatmap_3 = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        box_target_2 = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        box_target_3 = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))
        reg_weight_2 = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))
        reg_weight_3 = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0, max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0, max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        h_radiuses_alpha_2 = (feat_hs / 2. * self.alpha_2).int()
        w_radiuses_alpha_2 = (feat_ws / 2. * self.alpha_2).int()
        h_radiuses_alpha_3= (feat_hs / 2. * self.alpha_3).int()
        w_radiuses_alpha_3 = (feat_ws / 2. * self.alpha_3).int()

        # if self.alpha != self.beta:
        #     h_radiuses_beta = (feat_hs / 2. * self.beta).int()
        #     w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.wh_gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() / self.down_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            fake_heatmap_2 = fake_heatmap_2.zero_()
            self.draw_truncate_gaussian(fake_heatmap_2, ct_ints[k],
                                        h_radiuses_alpha_2[k].item(), w_radiuses_alpha_2[k].item())
            heatmap_2[cls_id] = torch.max(heatmap_2[cls_id], fake_heatmap_2)

            fake_heatmap_3 = fake_heatmap_3.zero_()
            self.draw_truncate_gaussian(fake_heatmap_3, ct_ints[k],
                                        h_radiuses_alpha_3[k].item(), w_radiuses_alpha_3[k].item())
            heatmap_3[cls_id] = torch.max(heatmap_3[cls_id], fake_heatmap_3)

            # if self.alpha != self.beta:
            #     fake_heatmap = fake_heatmap.zero_()
            #     self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
            #                                 h_radiuses_beta[k].item(), w_radiuses_beta[k].item())


            if self.wh_gaussian:
                box_target_inds = fake_heatmap > 0
                box_target_inds_2 = fake_heatmap_2 > 0
                box_target_inds_3 = fake_heatmap_3 > 0

            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                box_target_2[:, box_target_inds_2] = gt_boxes[k][:, None]
                box_target_3[:, box_target_inds_3] = gt_boxes[k][:, None]
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]

            local_heatmap_2 = fake_heatmap_2[box_target_inds_2]
            ct_div_2 = local_heatmap_2.sum()
            local_heatmap_2 *= boxes_area_topk_log[k]

            local_heatmap_3 = fake_heatmap_3[box_target_inds_3]
            ct_div_3 = local_heatmap_3.sum()
            local_heatmap_3 *= boxes_area_topk_log[k]

            if self.wh_agnostic:
                cls_id = 0
            reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            reg_weight_2[cls_id, box_target_inds_2] = local_heatmap_2 / ct_div_2
            reg_weight_3[cls_id, box_target_inds_3] = local_heatmap_3 / ct_div_3

        return heatmap, box_target, reg_weight, heatmap_2, box_target_2, reg_weight_2, heatmap_3, box_target_3, reg_weight_3

    def target_generator(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, reg_weight, heatmap_2, box_target_2, reg_weight_2, heatmap_3, box_target_3, reg_weight_3 = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()

            heatmap_2, box_target_2 = [torch.stack(t, dim=0).detach() for t in [heatmap_2, box_target_2]]
            reg_weight_2 = torch.stack(reg_weight_2, dim=0).detach()

            heatmap_3, box_target_3 = [torch.stack(t, dim=0).detach() for t in [heatmap_3, box_target_3]]
            reg_weight_3 = torch.stack(reg_weight_3, dim=0).detach()

            return heatmap, box_target, reg_weight, heatmap_2, box_target_2, reg_weight_2, heatmap_3, box_target_3, reg_weight_3

    def loss_calc(self,
                  pred_hm,
                  pred_wh,
                  pred_hm_2,
                  pred_wh_2,
                  pred_hm_3,
                  pred_wh_3,
                  heatmap,
                  box_target,
                  wh_weight, heatmap_2, box_target_2, wh_weight_2, heatmap_3, box_target_3, wh_weight_3):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        pred_hm_2 = torch.clamp(pred_hm_2.sigmoid_(), min=1e-4, max=1 - 1e-4)
        pred_hm_3 = torch.clamp(pred_hm_3.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight
        hm_loss_2 = ct_focal_loss(pred_hm_2, heatmap_2) * self.hm_weight_2
        hm_loss_3 = ct_focal_loss(pred_hm_3, heatmap_3) * self.hm_weight_3

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        mask2 = wh_weight_2.view(-1, H, W)
        avg_factor2 = mask2.sum() + 1e-4

        mask3 = wh_weight_3.view(-1, H, W)
        avg_factor3 = mask3.sum() + 1e-4





        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        pred_boxes_2 = torch.cat((self.base_loc - pred_wh_2[:, [0, 1]],
                                self.base_loc + pred_wh_2[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        pred_boxes_3 = torch.cat((self.base_loc - pred_wh_3[:, [0, 1]],
                                  self.base_loc + pred_wh_3[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        boxes_2 = box_target_2.permute(0, 2, 3, 1)
        boxes_3 = box_target_3.permute(0, 2, 3, 1)
        wh_loss = giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) * self.wh_weight
        wh_loss_2 = giou_loss(pred_boxes_2, boxes_2, mask2, avg_factor=avg_factor2) * self.wh_weight_2
        wh_loss_3 = giou_loss(pred_boxes_3, boxes_3, mask3, avg_factor=avg_factor3) * self.wh_weight_3

        return hm_loss, wh_loss, hm_loss_2, wh_loss_2,hm_loss_3, wh_loss_3


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
