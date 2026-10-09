"""Single-file HiF4 submission using public eager PyTorch operations.

The six public interfaces are grouped at the end of this file. Shared encoding,
Linear calibration/online quantization, and Attention calibration/online
quantization are kept in separate sections below.

Numerical order is intentional: calibration and online Linear writers use
different reductions and sign handling and must not be interchanged. Frozen
state layouts, Q2/K2 iterations, and the 32768-block Linear budget are preserved.
"""
import math

import torch



# Fixed submission configuration


_CANDIDATE_OFFSETS = tuple(range(-4, 5))
_CHUNK_GROUPS = 16384
_QUANTIZE_CHUNK_GROUPS = 65536
_LINEAR_BLOCK_REFINE_STEPS = (0.04, 0.06, 0.08, 0.1, 0.12)
_ATTENTION_HEADWIDE_HEAD_DIM = 256
_RMS_EPSILON = 1e-12
_BALANCE_MIN = 2.0 ** (-8)
_BALANCE_MAX = 2.0 ** 8
_ENGINEERING_GROUPS = 4096
_CAMPAIGN_A_MODE = 'threshold104'
_LINEAR_A_PARENT_FACTOR = 1.0
_LINEAR_A_GROUPS = 4096
_LINEAR_RANK_STATIC_K = 16
_ATTN_LR_RANK = 3
_ATTN_Q_STEPS = 2
_ATTN_K_STEPS = 2


def _make_hadamard(size):
    if size <= 0 or size & size - 1:
        raise ValueError('Hadamard size must be a positive power of two')
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix.mul_(size ** (-0.5)).contiguous()


def _make_e6m2_table():
    values = [math.ldexp(1.0 + mantissa_code * 0.25, exponent) for exponent in range(-48, 16) for mantissa_code in range(4) if math.ldexp(1.0 + mantissa_code * 0.25, exponent) <= 49152.0]
    return torch.tensor(values, dtype=torch.float32)


_HADAMARD64 = _make_hadamard(64)
_HADAMARD256 = _make_hadamard(_ATTENTION_HEADWIDE_HEAD_DIM)
_E6M2 = _make_e6m2_table()
_ROOT_OFFSETS_TENSOR = torch.tensor(_CANDIDATE_OFFSETS, dtype=torch.int64)


# ============================================================================
# Shared HiF4 encoding and numerical utilities
# ============================================================================


def _hadamard64(values):
    if values.device.type != 'cpu':
        raise ValueError('contest tensors must be on CPU')
    if values.shape[-1] % 64:
        raise ValueError('last dimension must be divisible by 64')
    groups = values.to(torch.float32).reshape(-1, 64)
    return torch.matmul(groups, _HADAMARD64).reshape(values.shape)


def dequantize_nvfp4(quant_float, scale_float, blk_size=16):
    channels = int(quant_float.shape[-1])
    if channels % blk_size:
        raise ValueError(f'last dimension {channels} is not divisible by {blk_size}')
    expected = tuple(quant_float.shape[:-1]) + (channels // blk_size,)
    if tuple(scale_float.shape) != expected:
        raise ValueError(f'scale shape {tuple(scale_float.shape)} != {expected}')
    values = quant_float.unflatten(-1, (-1, blk_size))
    values = values * scale_float.unsqueeze(-1)
    return values.flatten(-2, -1).to(torch.bfloat16)


def _normalize_offsets(candidate_offsets):
    offsets = tuple(sorted({int(offset) for offset in candidate_offsets}))
    if not offsets:
        raise ValueError('candidate_offsets must not be empty')
    return offsets


def _nearest_e6m2_index(target):
    right = torch.searchsorted(_E6M2, target.contiguous()).clamp_max_(_E6M2.numel() - 1)
    left = (right - 1).clamp_min_(0)
    left_distance = (target - _E6M2[left]).abs()
    right_distance = (_E6M2[right] - target).abs()
    right_is_even = right.remainder(4).remainder(2) == 0
    choose_right = (right_distance < left_distance) | (right_distance == left_distance) & right_is_even
    return torch.where(choose_right, right, left)


def _engineering_nearest(target):
    mantissa, exponent = torch.frexp(target.clamp(2.0 ** (-48), 49152.0))
    return ((exponent + 47) * 4 + torch.round(mantissa * 8).to(torch.int32) - 4).clamp(0, 254).to(torch.int64)


def _mantissa_error(absolute, group_scale, multiplier, importance=None):
    scale = group_scale[:, None, None, None] * multiplier
    code = torch.round(absolute * (4.0 / scale)).clamp_(0.0, 7.0)
    squared_error = code.mul_(0.25 * scale).sub_(absolute).square_()
    if importance is not None:
        squared_error.mul_(importance)
    return squared_error.sum(dim=-1)


def _fixed_scale_tree(absolute, group_scale, *, return_params, importance=None):
    error_1 = _mantissa_error(absolute, group_scale, 1.0, importance)
    error_2 = _mantissa_error(absolute, group_scale, 2.0, importance)
    error_4 = _mantissa_error(absolute, group_scale, 4.0, importance)
    cost_lv2_1 = torch.minimum(error_1, error_2).sum(dim=-1)
    cost_lv2_2 = torch.minimum(error_2, error_4).sum(dim=-1)
    lv2_is_2 = cost_lv2_2 < cost_lv2_1
    total_error = torch.where(lv2_is_2, cost_lv2_2, cost_lv2_1).sum(dim=-1)
    if not return_params:
        return total_error
    lv3_when_lv2_1 = error_2 < error_1
    lv3_when_lv2_2 = error_4 < error_2
    lv3_is_2 = torch.where(lv2_is_2[..., None], lv3_when_lv2_2, lv3_when_lv2_1)
    multiplier = (1 + lv2_is_2.to(torch.int8))[..., None] * (1 + lv3_is_2.to(torch.int8))
    leaf_scale = group_scale[:, None, None, None] * multiplier[..., None]
    mantissa_code = torch.round(absolute * (4.0 / leaf_scale)).clamp_(0.0, 7.0)
    return (total_error, lv2_is_2, lv3_is_2, mantissa_code)


def _choose_group_scale(absolute, candidate_offsets, importance=None, reuse=None, *, max_abs=None, return_tree=False):
    if max_abs is None:
        max_abs = absolute.amax(dim=(-1, -2, -3))
    center = _nearest_e6m2_index(max_abs * (1.0 / 7.0))
    offsets = _normalize_offsets(candidate_offsets)
    if (importance is not None if reuse is None else reuse) and (importance is not None or absolute.shape[0] > 32) and (offsets == _CANDIDATE_OFFSETS) and bool(((center >= 4) & (center + 12 < _E6M2.numel())).all()):
        best_error = torch.full_like(max_abs, torch.inf)
        best_index = center.clone()
        if return_tree:
            best_lv2 = torch.zeros((len(absolute), 8), dtype=torch.bool)
            best_lv3 = torch.zeros((len(absolute), 8, 2), dtype=torch.bool)
            center_lv2 = None
            center_lv3 = None
        leaf_window = []
        for leaf_offset in range(-4, 13):
            leaf_window.append(_mantissa_error(absolute, _E6M2[center + leaf_offset], 1.0, importance))
            if len(leaf_window) < 9:
                continue
            error_1, error_2, error_4 = (leaf_window[0], leaf_window[4], leaf_window[8])
            low = torch.minimum(error_1, error_2).sum(dim=-1)
            high = torch.minimum(error_2, error_4).sum(dim=-1)
            candidate_lv2 = high < low
            candidate_error = torch.where(candidate_lv2, high, low).sum(dim=-1)
            offset = offsets[leaf_offset - 4]
            if return_tree:
                candidate_lv3 = torch.where(candidate_lv2[..., None], error_4 < error_2, error_2 < error_1)
                if offset == 0:
                    center_lv2 = candidate_lv2
                    center_lv3 = candidate_lv3
            improve = candidate_error < best_error
            best_error = torch.where(improve, candidate_error, best_error)
            best_index = torch.where(improve, center + offset, best_index)
            if return_tree:
                best_lv2 = torch.where(improve[..., None], candidate_lv2, best_lv2)
                best_lv3 = torch.where(improve[..., None, None], candidate_lv3, best_lv3)
            leaf_window.pop(0)
        group_scale = _E6M2[best_index]
        if return_tree:
            missing = torch.isinf(best_error)
            best_lv2 = torch.where(missing[..., None], center_lv2, best_lv2)
            best_lv3 = torch.where(missing[..., None, None], center_lv3, best_lv3)
            return (group_scale, best_lv2, best_lv3)
        return group_scale
    if absolute.shape[0] <= 32 and importance is None and (offsets == _CANDIDATE_OFFSETS):
        indexes = (center[:, None] + _ROOT_OFFSETS_TENSOR[None, :]).clamp(0, _E6M2.numel() - 1)
        expanded = absolute[:, None].expand(-1, 9, -1, -1, -1).reshape(-1, 8, 2, 4)
        costs = _fixed_scale_tree(expanded, _E6M2[indexes.reshape(-1)], return_params=False).reshape(absolute.shape[0], 9)
        winner = costs.argmin(dim=1, keepdim=True)
        index = indexes.gather(1, winner).squeeze(1)
        index = torch.where(torch.isfinite(costs.gather(1, winner).squeeze(1)), index, center)
        group_scale = _E6M2[index]
        return (group_scale, None, None) if return_tree else group_scale
    best_error = torch.full_like(max_abs, torch.inf)
    best_index = center.clone()
    for offset in offsets:
        index = (center + offset).clamp(0, _E6M2.numel() - 1)
        candidate_error = _fixed_scale_tree(absolute, _E6M2[index], return_params=False, importance=importance)
        improve = candidate_error < best_error
        best_error = torch.where(improve, candidate_error, best_error)
        best_index = torch.where(improve, index, best_index)
    group_scale = _E6M2[best_index]
    return (group_scale, None, None) if return_tree else group_scale


def _quantize_fp32_groups(values, candidate_offsets, importance=None, reuse=None, *, importance_validated=False):
    if values.dtype != torch.float32 or values.ndim != 2 or values.shape[1] != 64:
        raise ValueError('values must be FP32 [groups, 64]')
    if importance is not None:
        if importance.dtype != torch.float32 or importance.ndim != 2 or tuple(importance.shape) != tuple(values.shape):
            raise ValueError('importance must be FP32 [groups, 64]')
        if not importance_validated and (not torch.isfinite(importance).all() or bool((importance <= 0).any())):
            raise ValueError('importance must be finite and positive')
        importance = importance.reshape(-1, 8, 2, 4)
    absolute = values.abs().reshape(-1, 8, 2, 4)
    max_abs = absolute.amax(dim=(-1, -2, -3))
    if not torch.isfinite(max_abs).all():
        raise ValueError('HiF4 input contains non-finite values')
    group_scale, lv2_is_2, lv3_is_2 = _choose_group_scale(absolute, candidate_offsets, importance, reuse, max_abs=max_abs, return_tree=True)
    if lv2_is_2 is None:
        _, lv2_is_2, lv3_is_2, mantissa_code = _fixed_scale_tree(absolute, group_scale, return_params=True, importance=importance)
    else:
        multiplier = (1 + lv2_is_2.to(torch.int8))[..., None] * (1 + lv3_is_2.to(torch.int8))
        leaf_scale = group_scale[:, None, None, None] * multiplier[..., None]
        mantissa_code = torch.round(absolute * (4.0 / leaf_scale)).clamp_(0.0, 7.0)
    sign = torch.sign(values).reshape(-1, 8, 2, 4)
    sign.masked_fill_(mantissa_code == 0, 0)
    return (group_scale[:, None, None, None], (1 + lv2_is_2.to(torch.int8))[:, :, None, None], (1 + lv3_is_2.to(torch.int8))[:, :, :, None], sign, mantissa_code * 0.25)


def _allocate_outputs(group_count):
    dtype = torch.bfloat16
    return (torch.empty((group_count, 1, 1, 1), dtype=dtype), torch.empty((group_count, 8, 1, 1), dtype=torch.int8), torch.empty((group_count, 8, 2, 1), dtype=torch.int8), torch.empty((group_count, 8, 2, 4), dtype=torch.int8), torch.empty((group_count, 8, 2, 4), dtype=dtype))


def _format_outputs(outputs, prefix):
    scale_factor, scale_lv2, scale_lv3, sign, mantissa = outputs
    return {'scale_factor': scale_factor.reshape(prefix + (1, 1, 1)), 'scale_lv2': scale_lv2.reshape(prefix + (8, 1, 1)), 'scale_lv3': scale_lv3.reshape(prefix + (8, 2, 1)), 'sign': sign.reshape(prefix + (8, 2, 4)), 'mant': mantissa.reshape(prefix + (8, 2, 4))}


def _copy_chunk(outputs, start, end, params):
    for destination, source in zip(outputs, params):
        destination[start:end].copy_(source)


def _quantize_compact_row_importance(values, importance, offsets, chunk_groups, reuse):
    if values.ndim != 2 or importance.ndim != 1:
        return None
    groups_per_row = int(values.shape[-1]) // 64
    if importance.numel() != values.shape[-1] or offsets != _CANDIDATE_OFFSETS or (not (importance is not None if reuse is None else reuse)) or chunk_groups % groups_per_row:
        return None
    compact = importance.float().cpu().clamp_min(_RMS_EPSILON).reshape(groups_per_row, 8, 2, 4)
    if not torch.isfinite(compact).all():
        return None
    compact.div_(compact.reshape(groups_per_row, 64).mean(-1)[:, None, None, None])
    flat = values.reshape(-1, 64)
    outputs = _allocate_outputs(int(flat.shape[0]))
    for start in range(0, len(flat), chunk_groups):
        end = min(start + chunk_groups, len(flat))
        if start % groups_per_row or end % groups_per_row:
            return None
        rows = (end - start) // groups_per_row
        chunk = flat[start:end].float()
        absolute = chunk.abs().reshape(rows, groups_per_row, 8, 2, 4)
        max_abs = absolute.amax(dim=(-1, -2, -3))
        if not torch.isfinite(max_abs).all():
            return None
        center = _nearest_e6m2_index(max_abs * (1.0 / 7.0))
        if not bool(((center >= 4) & (center + 12 < _E6M2.numel())).all()):
            return None
        best_error = torch.full_like(max_abs, torch.inf)
        best_index = center.clone()
        best_lv2 = torch.zeros((rows, groups_per_row, 8), dtype=torch.bool)
        best_lv3 = torch.zeros((rows, groups_per_row, 8, 2), dtype=torch.bool)
        center_lv2 = center_lv3 = None
        leaf_window = []
        for leaf_offset in range(-4, 13):
            scale = _E6M2[center + leaf_offset][..., None, None, None]
            code = torch.round(absolute * (4.0 / scale)).clamp_(0.0, 7.0)
            error = code.mul_(0.25 * scale).sub_(absolute).square_()
            error.mul_(compact[None])
            leaf_window.append(error.sum(-1))
            if len(leaf_window) < 9:
                continue
            error_1, error_2, error_4 = (leaf_window[0], leaf_window[4], leaf_window[8])
            low = torch.minimum(error_1, error_2).sum(-1)
            high = torch.minimum(error_2, error_4).sum(-1)
            lv2 = high < low
            candidate_error = torch.where(lv2, high, low).sum(-1)
            offset = offsets[leaf_offset - 4]
            lv3 = torch.where(lv2[..., None], error_4 < error_2, error_2 < error_1)
            if offset == 0:
                center_lv2, center_lv3 = (lv2, lv3)
            improve = candidate_error < best_error
            torch.where(improve, candidate_error, best_error, out=best_error)
            best_index = torch.where(improve, center + offset, best_index)
            best_lv2 = torch.where(improve[..., None], lv2, best_lv2)
            best_lv3 = torch.where(improve[..., None, None], lv3, best_lv3)
            leaf_window.pop(0)
        missing = torch.isinf(best_error)
        best_lv2 = torch.where(missing[..., None], center_lv2, best_lv2)
        best_lv3 = torch.where(missing[..., None, None], center_lv3, best_lv3)
        root = _E6M2[best_index]
        multiplier = (1 + best_lv2.to(torch.int8))[..., None] * (1 + best_lv3.to(torch.int8))
        leaf_scale = root[..., None, None, None] * multiplier[..., None]
        mantissa = torch.round(absolute * (4.0 / leaf_scale)).clamp_(0.0, 7.0)
        sign = chunk.sign().reshape_as(mantissa)
        sign.masked_fill_(mantissa == 0.0, 0.0)
        _copy_chunk(outputs, start, end, (root.reshape(-1, 1, 1, 1), (1 + best_lv2.to(torch.int8)).reshape(-1, 8, 1, 1), (1 + best_lv3.to(torch.int8)).reshape(-1, 8, 2, 1), sign.reshape(-1, 8, 2, 4), mantissa.mul(0.25).reshape(-1, 8, 2, 4)))
    prefix = tuple(values.shape[:-1]) + (groups_per_row,)
    return _format_outputs(outputs, prefix)


@torch.inference_mode()
def _quantize_tensor(values, *, importance=None, candidate_offsets=_CANDIDATE_OFFSETS, chunk_groups=_QUANTIZE_CHUNK_GROUPS, reuse=None):
    if values.device.type != 'cpu':
        raise ValueError('contest tensors must be on CPU')
    if values.shape[-1] % 64:
        raise ValueError('last dimension must be divisible by 64')
    if chunk_groups <= 0:
        raise ValueError('chunk_groups must be positive')
    offsets = _normalize_offsets(candidate_offsets)
    if importance is not None:
        compact = _quantize_compact_row_importance(values, importance, offsets, chunk_groups, reuse)
        if compact is not None:
            return compact
    flat = values.reshape(-1, 64)
    group_count = flat.shape[0]
    outputs = _allocate_outputs(group_count)
    importance_groups = None
    groups_per_row = int(values.shape[-1]) // 64
    if importance is not None:
        if importance.ndim != 1 or importance.numel() != values.shape[-1]:
            raise ValueError('importance must match the last dimension')
        importance_groups = importance.float().cpu().clamp_min(_RMS_EPSILON).reshape(groups_per_row, 64)
        if not torch.isfinite(importance_groups).all():
            raise ValueError('importance must be finite')
        importance_groups.div_(importance_groups.mean(dim=-1, keepdim=True))
    for start in range(0, group_count, chunk_groups):
        end = min(start + chunk_groups, group_count)
        chunk_importance = None
        if importance_groups is not None:
            group_index = torch.arange(start, end).remainder_(groups_per_row)
            chunk_importance = importance_groups.index_select(0, group_index)
        params = _quantize_fp32_groups(flat[start:end].float(), offsets, chunk_importance, reuse, importance_validated=chunk_importance is not None)
        _copy_chunk(outputs, start, end, params)
    prefix = tuple(values.shape[:-1]) + (int(values.shape[-1]) // 64,)
    return _format_outputs(outputs, prefix)


def _reconstruct_hif4(params, shape):
    values = params['sign'].float() * params['mant'].float() * params['scale_lv3'].float() * params['scale_lv2'].float() * params['scale_factor'].float()
    return values.reshape(shape)


def _normalized_leaf_error(scaled_absolute, importance, multiplier_squared):
    error = torch.round(scaled_absolute).clamp_(0.0, 7.0)
    error.sub_(scaled_absolute).square_()
    if importance is not None:
        error.mul_(importance)
    result = error.sum(dim=-1)
    if multiplier_squared != 1.0:
        result.mul_(multiplier_squared)
    return result


def _fixed_scale_tree_normalized(absolute, group_scale, *, return_params, importance=None, physical_cost=True, restore_physical_mantissa=False):
    """Same shapes as baseline, approximate FP32 objective; cost units explicit.

    For single-root callers that ignore returned cost use physical_cost=False.
    For candidate-root selection physical_cost must be True.
    """
    t = absolute * (4.0 / group_scale)[:, None, None, None]
    error_1 = _normalized_leaf_error(t, importance, 1.0)
    error_2 = _normalized_leaf_error(t * 0.5, importance, 4.0)
    error_4 = _normalized_leaf_error(t * 0.25, importance, 16.0)
    low = torch.minimum(error_1, error_2).sum(dim=-1)
    high = torch.minimum(error_2, error_4).sum(dim=-1)
    lv2 = high < low
    total_error = torch.where(lv2, high, low).sum(dim=-1)
    if physical_cost:
        root_step = group_scale * 0.25
        total_error.mul_(root_step.square())
    if not return_params:
        return total_error
    lv3 = torch.where(lv2[..., None], error_4 < error_2, error_2 < error_1)
    multiplier = (1 + lv2.to(torch.int8))[..., None] * (1 + lv3.to(torch.int8))
    if restore_physical_mantissa:
        leaf_scale = group_scale[:, None, None, None] * multiplier[..., None]
        mant = torch.round(absolute * (4.0 / leaf_scale)).clamp_(0.0, 7.0)
    else:
        mant = torch.round(t * multiplier[..., None].to(torch.float32).reciprocal()).clamp_(0.0, 7.0)
    return (total_error, lv2, lv3, mant)


def _safe_balance(numerator_rms, denominator_rms, alpha):
    ratio = numerator_rms.clamp_min(_RMS_EPSILON) / denominator_rms.clamp_min(_RMS_EPSILON)
    return ratio.pow(alpha).clamp_(_BALANCE_MIN, _BALANCE_MAX).float().cpu()


# ============================================================================
# Linear calibration: transforms, weight refinement, and frozen state
# ============================================================================


def _refine_hif4_mantissa_blocks(target, params, block_gram, *, step_sizes=_LINEAR_BLOCK_REFINE_STEPS, weight_coordinate_steps=0, activation_coordinate_steps=0, reuse_winner_gradient=False):
    if target.device.type != 'cpu' or target.dtype != torch.float32:
        raise ValueError('block refinement target must be an FP32 CPU tensor')
    if target.ndim != 2 or target.shape[-1] % 64:
        raise ValueError('block refinement target must be 2-D with 64-aligned width')
    blocks_per_row = int(target.shape[-1]) // 64
    if block_gram.device.type != 'cpu' or block_gram.dtype != torch.float32 or tuple(block_gram.shape) != (blocks_per_row, 64, 64):
        raise ValueError('block Gram must be FP32 [width/64, 64, 64] on CPU')
    if not torch.isfinite(block_gram).all():
        raise ValueError('block Gram must be finite')
    if weight_coordinate_steps not in (0, 4):
        raise ValueError('weight coordinate refinement supports zero or four steps')
    if activation_coordinate_steps not in (0, 1, 2):
        raise ValueError('activation coordinate refinement supports zero, one or two steps')
    if weight_coordinate_steps and activation_coordinate_steps:
        raise ValueError('weight and activation coordinate refinement are exclusive')
    steps = tuple((float(step) for step in step_sizes))
    if not steps or any((not math.isfinite(step) or not 0.0 <= step <= 1.0 for step in steps)):
        raise ValueError('block refinement steps must be finite and in [0, 1]')
    keep_gradient = bool(reuse_winner_gradient and (weight_coordinate_steps or activation_coordinate_steps))
    diagonal = block_gram.diagonal(dim1=-2, dim2=-1)
    active = diagonal > _RMS_EPSILON
    safe_diagonal = diagonal.clamp_min(_RMS_EPSILON)
    result = dict(params)
    result['sign'] = params['sign'].clone()
    result['mant'] = params['mant'].clone()
    rows = int(target.shape[0])
    rows_per_chunk = max(1, _CHUNK_GROUPS // blocks_per_row)
    parameter_rows = {name: value.reshape(rows, blocks_per_row, *value.shape[2:]) for name, value in params.items()}
    sign_output = result['sign'].reshape_as(parameter_rows['sign'])
    mantissa_output = result['mant'].reshape_as(parameter_rows['mant'])
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        target_blocks = target[start:end].reshape(-1, blocks_per_row, 64)
        quant_step = (parameter_rows['scale_factor'][start:end].float() * parameter_rows['scale_lv2'][start:end].float() * parameter_rows['scale_lv3'][start:end].float() * 0.25).expand_as(parameter_rows['sign'][start:end])
        quant_step = quant_step.reshape_as(target_blocks)
        old_code = (parameter_rows['sign'][start:end].float() * parameter_rows['mant'][start:end].float() * 4.0).reshape_as(target_blocks)
        error = target_blocks - old_code * quant_step
        gradient = torch.einsum('rbi,bij->rbj', error, block_gram)
        best_cost = (error * gradient).sum(dim=-1)
        error.mul_(diagonal)
        correction = torch.sub(gradient, error).div_(safe_diagonal)
        correction.masked_fill_(~active.unsqueeze(0), 0.0)
        best_code = old_code
        best_gradient = gradient if keep_gradient else None
        for step in steps:
            candidate_code = torch.mul(correction, step)
            candidate_code.add_(target_blocks)
            candidate_code.div_(quant_step).round_().clamp_(-7.0, 7.0)
            candidate_error = target_blocks - candidate_code * quant_step
            candidate_gradient = torch.einsum('rbi,bij->rbj', candidate_error, block_gram)
            candidate_cost = candidate_error.mul_(candidate_gradient).sum(dim=-1)
            improve = candidate_cost < best_cost
            torch.where(improve, candidate_cost, best_cost, out=best_cost)
            torch.where(improve.unsqueeze(-1), candidate_code, best_code, out=best_code)
            if keep_gradient:
                torch.where(improve.unsqueeze(-1), candidate_gradient, best_gradient, out=best_gradient)
        if weight_coordinate_steps:
            best_code = _refine_weight_coordinate_codes(target_blocks, best_code, quant_step, block_gram, diagonal, safe_diagonal, active, weight_coordinate_steps, initial_gradient=best_gradient, skip_final_update=keep_gradient)
        elif activation_coordinate_steps:
            if not keep_gradient:
                best_gradient = torch.einsum('rbi,bij->rbj', target_blocks - best_code * quant_step, block_gram)
            best_code = _refine_activation_coordinate_code(best_code, quant_step, diagonal, safe_diagonal, active, best_gradient, block_gram, activation_coordinate_steps)
        code = best_code.reshape_as(parameter_rows['sign'][start:end])
        sign_output[start:end].copy_(torch.sign(code))
        mantissa_output[start:end].copy_(code.abs().mul_(0.25))
    return result


def _refine_weight_coordinate_codes(values, code, quant_step, block_gram, diagonal, safe_diagonal, active, iterations, *, initial_gradient=None, skip_final_update=False):
    quantized = code * quant_step
    gradient = torch.einsum('rbi,bij->rbj', values - quantized, block_gram) if initial_gradient is None else initial_gradient
    rows_count, blocks, dim = code.shape
    gram_rows = block_gram.reshape(-1, dim)
    offsets = torch.arange(blocks) * dim
    diagonal_view = diagonal.unsqueeze(0)
    safe_view = safe_diagonal.unsqueeze(0)
    inactive_view = ~active.unsqueeze(0)
    for iteration in range(iterations):
        candidate_code = torch.round((quantized + gradient / safe_view) / quant_step).clamp_(-7.0, 7.0)
        delta = candidate_code * quant_step - quantized
        delta_cost = delta.square() * diagonal_view - 2.0 * delta * gradient
        delta_cost.masked_fill_(inactive_view, torch.inf)
        best_cost, index = delta_cost.min(dim=-1, keepdim=True)
        improve = best_cost < 0
        chosen_code = torch.where(improve, candidate_code.gather(-1, index), code.gather(-1, index))
        code.scatter_(-1, index, chosen_code)
        if skip_final_update and iteration == iterations - 1:
            continue
        change = torch.where(improve, delta.gather(-1, index), 0.0)
        flat_index = (index.squeeze(-1) + offsets).reshape(-1)
        selected_gram_row = gram_rows.index_select(0, flat_index).reshape(rows_count, blocks, dim)
        gradient.sub_(change * selected_gram_row)
        quantized.scatter_(-1, index, chosen_code * quant_step.gather(-1, index))
    return code


def _refine_activation_coordinate_code(code, quant_step, diagonal, safe_diagonal, active, gradient, block_gram=None, iterations=1):
    if iterations not in (1, 2):
        raise ValueError('activation coordinate refinement supports one or two steps')
    if iterations > 1 and block_gram is None:
        raise ValueError('multi-step activation refinement requires block Gram')
    quantized = code * quant_step
    rows_count, blocks, dim = code.shape
    offsets = torch.arange(blocks) * dim
    gram_rows = None if block_gram is None else block_gram.reshape(-1, dim)
    for iteration in range(iterations):
        candidate_code = torch.round((quantized + gradient / safe_diagonal) / quant_step).clamp_(-7.0, 7.0)
        delta = candidate_code * quant_step - quantized
        delta_cost = delta.square() * diagonal - 2.0 * delta * gradient
        delta_cost.masked_fill_(~active.unsqueeze(0), torch.inf)
        best_cost, index = delta_cost.min(dim=-1, keepdim=True)
        improve = best_cost < 0
        chosen_code = torch.where(improve, candidate_code.gather(-1, index), code.gather(-1, index))
        code.scatter_(-1, index, chosen_code)
        if iteration == iterations - 1:
            continue
        change = torch.where(improve, delta.gather(-1, index), 0.0)
        flat_index = (index.squeeze(-1) + offsets).reshape(-1)
        selected_gram_row = gram_rows.index_select(0, flat_index).reshape(rows_count, blocks, dim)
        gradient.sub_(change * selected_gram_row)
        quantized.scatter_(-1, index, chosen_code * quant_step.gather(-1, index))
    return code


def _finalize_block_gram_(block_gram, sample_count):
    if sample_count <= 0:
        raise ValueError('block Gram sample count must be positive')
    block_gram.div_(sample_count)
    block_gram.copy_((block_gram + block_gram.transpose(-1, -2)).mul_(0.5))
    mean_diagonal = block_gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    block_gram.div_(mean_diagonal.clamp_min(_RMS_EPSILON)[:, None, None])


def _linear_matching_two_exchange(cost):
    n = len(cost)
    work = cost.clone()
    work.diagonal().fill_(torch.inf)
    pairs = []
    for _ in range(n // 2):
        i, j = divmod(int(work.argmin()), n)
        pairs.append([i, j])
        work[i, :] = torch.inf
        work[:, i] = torch.inf
        work[j, :] = torch.inf
        work[:, j] = torch.inf
    c = (cost.double() + cost.T.double()) * 0.5
    initial = sum((float(c[i, j]) for i, j in pairs))
    upper = torch.ones(len(pairs), len(pairs), dtype=torch.bool).triu(1)
    for _ in range(32):
        p = torch.tensor(pairs)
        a, b = (p[:, 0], p[:, 1])
        old = c[a, b]
        total = old[:, None] + old[None]
        gains = torch.stack([total - c[a[:, None], a[None]] - c[b[:, None], b[None]], total - c[a[:, None], b[None]] - c[b[:, None], a[None]]])
        gains[:, ~upper] = -torch.inf
        gain, idx = gains.flatten().max(0)
        if float(gain) <= 1e-12 * (1 + abs(initial)):
            break
        option, remainder = divmod(int(idx), len(pairs) ** 2)
        i, j = divmod(remainder, len(pairs))
        a, b = pairs[i]
        c1, d = pairs[j]
        if option == 0:
            pairs[i], pairs[j] = (sorted([a, c1]), sorted([b, d]))
        else:
            pairs[i], pairs[j] = (sorted([a, d]), sorted([b, c1]))
    return pairs


def _linear_magnitude_tree_permutation(weight, activations):
    blocks = int(weight.shape[-1]) // 64
    distance = torch.zeros((blocks, 64, 64), dtype=torch.float32)
    for values in (weight, torch.cat(activations, dim=0)):
        magnitude = values.abs().reshape(-1, blocks, 64)
        rms = magnitude.square().mean(dim=0).sqrt().clamp_min(1e-12)
        magnitude = magnitude / rms.square().mean(dim=-1, keepdim=True).sqrt()[None]
        second = magnitude.square().mean(dim=0)
        cross = torch.einsum('rbi,rbj->bij', magnitude, magnitude) / len(magnitude)
        distance.add_((second[:, :, None] + second[:, None, :] - 2 * cross).clamp_min(0), alpha=0.5)
    permutation = []
    for block in range(blocks):
        clusters = [[i] for i in range(64)]
        for _ in (2, 4, 8):
            index = torch.tensor(clusters, dtype=torch.int64)
            cost = distance[block][index[:, None, :, None], index[None, :, None, :]].mean((-1, -2))
            pairs = _linear_matching_two_exchange(cost)
            clusters = [clusters[left] + clusters[right] for left, right in pairs]
        permutation.extend((channel + block * 64 for group in clusters for channel in group))
    return torch.tensor(permutation, dtype=torch.int64)


def _ab_wide_accumulate(destination, values):
    size = destination.shape[-1]
    x = values.reshape(len(values), -1, size).transpose(0, 1)
    destination.add_(x.transpose(1, 2) @ x)


@torch.inference_mode()
def _ab_wide_refine(target, params, gram, *, steps, iterations):
    size = int(gram.shape[-1])
    rows, width = target.shape
    groups = width // size
    small_groups = width // 64
    if width % size or tuple(gram.shape) != (groups, size, size):
        raise ValueError('wide Gram shape mismatch')
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    active = diagonal > _RMS_EPSILON
    safe_diagonal = diagonal.clamp_min(_RMS_EPSILON)
    result = dict(params)
    result['sign'] = params['sign'].clone()
    result['mant'] = params['mant'].clone()
    parameter_rows = {name: value.reshape(rows, small_groups, *value.shape[2:]) for name, value in params.items()}
    sign_output = result['sign'].reshape_as(parameter_rows['sign'])
    mantissa_output = result['mant'].reshape_as(parameter_rows['mant'])
    rows_per_chunk = max(1, _CHUNK_GROUPS // small_groups)
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        values = target[start:end].reshape(-1, groups, size)
        quant_step = parameter_rows['scale_factor'][start:end].float() * parameter_rows['scale_lv2'][start:end].float() * parameter_rows['scale_lv3'][start:end].float() * 0.25
        quant_step = quant_step.expand_as(parameter_rows['sign'][start:end]).reshape_as(values)
        code = (parameter_rows['sign'][start:end].float() * parameter_rows['mant'][start:end].float() * 4).reshape_as(values)
        error = values - code * quant_step
        gradient = torch.einsum('rbi,bij->rbj', error, gram)
        if steps:
            correction = (gradient - error * diagonal) / safe_diagonal
            correction.masked_fill_(~active.unsqueeze(0), 0)
            best_cost = (error * gradient).sum(-1)
            for step in steps:
                candidate = torch.round((values + step * correction) / quant_step).clamp_(-7, 7)
                candidate_error = values - candidate * quant_step
                candidate_gradient = torch.einsum('rbi,bij->rbj', candidate_error, gram)
                candidate_cost = (candidate_error * candidate_gradient).sum(-1)
                improve = candidate_cost < best_cost
                best_cost = torch.where(improve, candidate_cost, best_cost)
                code = torch.where(improve.unsqueeze(-1), candidate, code)
                torch.where(improve.unsqueeze(-1), candidate_gradient, gradient, out=gradient)
        quantized = code * quant_step
        for iteration in range(iterations):
            candidate = torch.round((quantized + gradient / safe_diagonal) / quant_step).clamp_(-7, 7)
            delta = candidate * quant_step - quantized
            delta_cost = delta.square() * diagonal - 2 * delta * gradient
            delta_cost.masked_fill_(~active.unsqueeze(0), torch.inf)
            best_cost, index = delta_cost.min(-1, keepdim=True)
            improve = best_cost < 0
            chosen = torch.where(improve, candidate.gather(-1, index), code.gather(-1, index))
            code.scatter_(-1, index, chosen)
            if iteration + 1 < iterations:
                change = torch.where(improve, delta.gather(-1, index), 0.0)
                flat_index = (index.squeeze(-1) + torch.arange(groups) * size).reshape(-1)
                gram_row = gram.reshape(-1, size).index_select(0, flat_index).reshape(values.shape[0], groups, size)
                gradient.sub_(change * gram_row)
                quantized.scatter_(-1, index, chosen * quant_step.gather(-1, index))
        code = code.reshape_as(parameter_rows['sign'][start:end])
        sign_output[start:end].copy_(torch.sign(code))
        mantissa_output[start:end].copy_(code.abs().mul_(0.25))
    return result


def _linear_energy_spread_permutation(weight_rms, activation_rms):
    order = torch.argsort((weight_rms * activation_rms).sqrt(), stable=True)
    batches = order.reshape(64, -1).clone()
    batches[1::2] = batches[1::2].flip(1)
    return batches.T.contiguous().reshape(-1)


def _finals_linear_permutation(local):
    width = local.numel()
    return (local.to(torch.int64).reshape(-1, 64) + torch.arange(width // 64, dtype=torch.int64)[:, None] * 64).reshape(-1)


def _linear_pack_local(local):
    values = local.to(torch.int32).reshape(-1, 4)
    word = values[:, 0] | values[:, 1] << 6 | values[:, 2] << 12 | values[:, 3] << 18
    return torch.stack((word & 255, word >> 8 & 255, word >> 16 & 255), dim=-1).to(torch.int8).reshape(-1)


def _linear_unpack_local(packed):
    data = packed.to(torch.int64).reshape(-1, 3) & 255
    word = data[:, 0] | data[:, 1] << 8 | data[:, 2] << 16
    return torch.stack((word & 63, word >> 6 & 63, word >> 12 & 63, word >> 18 & 63), dim=-1).reshape(-1).to(torch.int8)


def _linear_rank_state(result, shape):
    balance, pre, local, importance = result['activation_state']
    local = _linear_pack_local(local)
    rank_code, rank_scale = (None, None)
    if importance is not None and shape[0] >= 4:
        weight = _reconstruct_hif4(result['weight_params'], shape)
        diagonal = weight.square().mean(0).clamp_min(1e-12)
        normalizer = diagonal.mean().clamp_min(1e-12)
        z = weight.div_(normalizer.sqrt())
        index = torch.arange(shape[-1], dtype=torch.float32) + 1
        directions = torch.stack([torch.sin(index * (j + 1)) for j in range(4)], dim=-1)
        directions = torch.linalg.qr(directions).Q
        for _ in range(12):
            directions = torch.linalg.qr(z.T @ (z @ directions) / len(z)).Q
        projected = z @ directions
        eigenvalues, rotation = torch.linalg.eigh(projected.T @ projected / len(z))
        vector = directions @ rotation[:, -1] * eigenvalues[-1].clamp_min(0).sqrt()
        rank_scale = float(vector.abs().max().clamp_min(1e-12) / 127.0)
        rank_code = torch.round(vector / rank_scale).clamp_(-127, 127).to(torch.int8)
        importance = (diagonal / normalizer).clamp_(0.0001, 65504).to(torch.float16)
    result['activation_state'] = (balance, pre, local, importance, rank_code, rank_scale)
    return result


def _linear_w_choose(target, gram, original, candidate):
    size = gram.shape[-1]
    costs = []
    for params in (original, candidate):
        error = (target - _reconstruct_hif4(params, target.shape)).reshape(len(target), -1, size)
        costs.append((error * torch.einsum('rbi,bij->rbj', error, gram)).sum(-1))
    improve = (costs[1] < costs[0]).repeat_interleave(size // 64, -1)
    return {key: torch.where(improve.reshape(*improve.shape, *[1] * (value.ndim - 2)), candidate[key], value) for key, value in original.items()}


def _linear_w_schur(target, gram):
    result = {}
    size = gram.shape[-1]
    eye = torch.eye(64, dtype=torch.float64)
    for group in range(len(gram)):
        metric = gram[group].double()
        ridge = metric.diagonal().mean().clamp_min(1e-20) * 1e-06
        metric = metric + torch.eye(size, dtype=torch.float64) * ridge
        upper = torch.linalg.solve_triangular(torch.linalg.cholesky(metric.flip((0, 1)), upper=False).flip((0, 1)), torch.eye(size, dtype=torch.float64), upper=True)
        work = target[:, group * size:(group + 1) * size].clone()
        for first in range(0, size, 64):
            inverse = torch.linalg.solve_triangular(upper[first:first + 64, first:first + 64], eye, upper=True)
            condition = (inverse @ inverse.T).float()
            values = work[:, first:first + 64].contiguous()
            params = _quantize_tensor(values, importance=condition.diagonal().clone(), reuse=True)
            params = _refine_hif4_mantissa_blocks(values, params, condition[None], weight_coordinate_steps=4)
            reconstructed = _reconstruct_hif4(params, values.shape)
            if first + 64 < size:
                error = (values - reconstructed) @ inverse.float()
                work[:, first + 64:].sub_(error @ upper[first:first + 64, first + 64:].float())
            for key, value in params.items():
                result.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=1) for key, values in result.items()}


def _refine_operator_operand(values, params, operand_state):
    gram = operand_state['block_gram']
    gram = 0.5 * gram + 0.5 * torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1))
    if gram.shape[-1] <= 64:
        return _refine_hif4_mantissa_blocks(values, params, gram, weight_coordinate_steps=4)
    candidate = _linear_w_schur(values, gram)
    candidate = _ab_wide_refine(values, candidate, gram, steps=_LINEAR_BLOCK_REFINE_STEPS, iterations=2 * (gram.shape[-1] // 64))
    return _linear_w_choose(values, gram, params, candidate)


def _goal_log_pack(values):
    logarithm = values.float().clamp_min(1e-12).log2()
    low,high = logarithm.min(),logarithm.max()
    step = ((high-low)/255.0).clamp_min(1e-12)
    code = torch.round((logarithm-low)/step).clamp_(0,255).to(torch.int16)
    code = (code-128).to(torch.int8)
    return code,torch.stack((step,low+128*step)).float()


def _goal_log_unpack(code,metadata):
    return torch.exp2(code.float()*metadata[0]+metadata[1])


def _goal_householder_normalize(vector):
    vector = vector.float()
    return vector/vector.norm(dim=-1,keepdim=True).clamp_min(1e-12)


def _goal_householder_apply(values,vector):
    for direction in vector:
        values=values-2.0*(values@direction).unsqueeze(-1)*direction
    return values


def _goal_householder_direction(samples,width):
    index = torch.arange(width,dtype=torch.float32)+1
    stored=[]
    for axis in range(2):
        direction = torch.sin(index*(axis+1))
        direction[:axis]=0
        direction.div_(direction.norm().clamp_min(1e-12))
        previous = None if not stored else _goal_householder_normalize(torch.stack(stored))
        for _ in range(12):
            moment = torch.zeros(width,dtype=torch.float32)
            for pair in samples:
                values = dequantize_nvfp4(*pair).float().reshape(-1,width)
                if previous is not None:
                    values = _goal_householder_apply(values,previous)
                moment.add_(values.T@(values@direction))
            # Fix already aligned axes exactly, including after INT8 decoding.
            moment[:axis]=0
            direction=moment/moment.norm().clamp_min(1e-12)
        if direction.norm()<1e-8:
            stored.append(torch.zeros(width,dtype=torch.int8))
            continue
        if direction[axis]>0:
            direction=-direction
        direction[axis]-=1.0
        maximum=direction.abs().max().clamp_min(1e-12)
        stored.append(torch.round(direction*(127.0/maximum)).clamp_(-127,127).to(torch.int8))
    return torch.stack(stored).contiguous()


@torch.inference_mode()
def _goal_original_calibration(weight_quant, weight_scale, calib_activation_list):
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    width = int(weight.shape[-1])
    if width <= 0 or width % 64:
        raise ValueError('Linear width must be a positive multiple of 64')
    weight_rms = weight.square().mean(0).sqrt() if len(weight) else torch.ones(width, dtype=torch.float32)
    square_sum = torch.zeros(width, dtype=torch.float64)
    count = 0
    for pair in calib_activation_list:
        activation = dequantize_nvfp4(*pair).float().reshape(-1, width)
        square_sum.add_(activation.square().sum(0, dtype=torch.float64))
        count += len(activation)
    activation_rms = (square_sum / count).sqrt().float() if count else torch.ones(width, dtype=torch.float32)
    balance = _safe_balance(weight_rms, activation_rms, 0.5).to(torch.float16)
    if width <= 32768:
        pre = _linear_energy_spread_permutation(weight_rms, activation_rms).to(torch.int16)
    elif width <= 2147483648:
        pre = _linear_energy_spread_permutation(weight_rms, activation_rms).to(torch.int32)
    else:
        pre = None
    weight.div_(balance.float())
    if pre is not None:
        weight = weight.index_select(-1, pre.to(torch.int64))
    weight = _hadamard64(weight)
    transformed = []
    size = width if width == 3072 else math.gcd(width, 2048)
    groups = width // size
    gram = torch.zeros((groups, size, size), dtype=torch.float32)
    for pair in calib_activation_list:
        activation = dequantize_nvfp4(*pair).float().reshape(-1, width)
        if not len(activation):
            continue
        activation.mul_(balance.float())
        if pre is not None:
            activation = activation.index_select(-1, pre.to(torch.int64))
        activation = _hadamard64(activation)
        _ab_wide_accumulate(gram, activation)
        transformed.append(activation)
    if len(weight) and transformed:
        permutation = _linear_magnitude_tree_permutation(weight, transformed)
    else:
        permutation = torch.arange(width, dtype=torch.int64)
    del transformed
    weight = weight.index_select(-1, permutation)
    within = permutation.reshape(groups, size).remainder(size)
    block = torch.arange(groups)[:, None, None]
    gram = gram[block, within[:, :, None], within[:, None, :]]
    if count:
        _finalize_block_gram_(gram, count)
    params = _quantize_tensor(weight, reuse=True)
    if count and len(weight):
        params = _refine_operator_operand(weight, params, {'block_gram': gram, 'weight_coordinate_steps': 4})
    importance = None
    if pre is None or pre.element_size() == 2:
        quantized = _reconstruct_hif4(params, weight.shape)
        importance = quantized.square().mean(0) if len(weight) else torch.ones(width, dtype=torch.float32)
        importance = importance.reshape(-1, 64).clamp_min(1e-12)
        importance.div_(importance.mean(-1, keepdim=True).clamp_min(1e-12))
        importance = importance.clamp_(0.0001, 65504).reshape(-1).to(torch.float16)
    local = permutation.remainder(64).to(torch.int8)
    return _linear_rank_state({'weight_params': params, 'activation_state': (balance.contiguous(), pre, local.contiguous(), importance)}, weight.shape)


@torch.inference_mode()
def _goal_householder_calibration(weight_quant, weight_scale, calib_activation_list, stored_vector):
    hh_vector = _goal_householder_normalize(stored_vector)
    weight = _goal_householder_apply(dequantize_nvfp4(weight_quant, weight_scale).float(),hh_vector)
    width = int(weight.shape[-1])
    if width <= 0 or width % 64:
        raise ValueError('Linear width must be a positive multiple of 64')
    weight_rms = weight.square().mean(0).sqrt() if len(weight) else torch.ones(width, dtype=torch.float32)
    square_sum = torch.zeros(width, dtype=torch.float64)
    count = 0
    for pair in calib_activation_list:
        activation = _goal_householder_apply(dequantize_nvfp4(*pair).float().reshape(-1,width),hh_vector)
        square_sum.add_(activation.square().sum(0, dtype=torch.float64))
        count += len(activation)
    activation_rms = (square_sum / count).sqrt().float() if count else torch.ones(width, dtype=torch.float32)
    balance_code,balance_metadata = _goal_log_pack(_safe_balance(weight_rms,activation_rms,0.5))
    balance = _goal_log_unpack(balance_code,balance_metadata)
    if width <= 32768:
        pre = _linear_energy_spread_permutation(weight_rms, activation_rms).to(torch.int16)
    elif width <= 2147483648:
        pre = _linear_energy_spread_permutation(weight_rms, activation_rms).to(torch.int32)
    else:
        pre = None
    weight.div_(balance.float())
    if pre is not None:
        weight = weight.index_select(-1, pre.to(torch.int64))
    weight = _hadamard64(weight)
    transformed = []
    size = width if width == 3072 else math.gcd(width, 2048)
    groups = width // size
    gram = torch.zeros((groups, size, size), dtype=torch.float32)
    for pair in calib_activation_list:
        activation = _goal_householder_apply(dequantize_nvfp4(*pair).float().reshape(-1,width),hh_vector)
        if not len(activation):
            continue
        activation.mul_(balance.float())
        if pre is not None:
            activation = activation.index_select(-1, pre.to(torch.int64))
        activation = _hadamard64(activation)
        _ab_wide_accumulate(gram, activation)
        transformed.append(activation)
    if len(weight) and transformed:
        permutation = _linear_magnitude_tree_permutation(weight, transformed)
    else:
        permutation = torch.arange(width, dtype=torch.int64)
    del transformed
    weight = weight.index_select(-1, permutation)
    within = permutation.reshape(groups, size).remainder(size)
    block = torch.arange(groups)[:, None, None]
    gram = gram[block, within[:, :, None], within[:, None, :]]
    if count:
        _finalize_block_gram_(gram, count)
    params = _quantize_tensor(weight, reuse=True)
    if count and len(weight):
        params = _refine_operator_operand(weight, params, {'block_gram': gram, 'weight_coordinate_steps': 4})
    importance = None
    if pre is None or pre.element_size() == 2:
        quantized = _reconstruct_hif4(params, weight.shape)
        importance = quantized.square().mean(0) if len(weight) else torch.ones(width, dtype=torch.float32)
        importance = importance.reshape(-1, 64).clamp_min(1e-12)
        importance.div_(importance.mean(-1, keepdim=True).clamp_min(1e-12))
        importance = importance.clamp_(0.0001, 65504).reshape(-1).to(torch.float16)
    local = permutation.remainder(64).to(torch.int8)
    result = _linear_rank_state({'weight_params':params,'activation_state':(balance.contiguous(),pre,local.contiguous(),importance)},weight.shape)
    state = list(result['activation_state'])
    diagonal_code,diagonal_metadata = _goal_log_pack(state[3].float())
    state[0] = balance_code
    state[3] = diagonal_code
    state.extend((stored_vector.contiguous(),torch.cat((balance_metadata,diagonal_metadata))))
    result['activation_state'] = tuple(state)
    return result


@torch.inference_mode()
def _combined_core_calibration(weight_quant,weight_scale,calib_activation_list):
    width = int(weight_quant.shape[-1])
    if width<1024 or width>32768 or not calib_activation_list:
        return _goal_original_calibration(weight_quant,weight_scale,calib_activation_list)
    vector = _goal_householder_direction(calib_activation_list,width)
    return _goal_householder_calibration(weight_quant,weight_scale,calib_activation_list,vector)


def _combined_policy_values(pair, width, balance, pre, local, hh_vector):
    count = pair[0].numel() // width
    if not count:
        return None
    indices = torch.linspace(0, count - 1, min(count, 128)).round().long()
    quant = pair[0].reshape(-1, width).index_select(0, indices)
    scale = pair[1].reshape(-1, width // 16).index_select(0, indices)
    values = dequantize_nvfp4(quant, scale).float()
    if hh_vector is not None:
        values = _goal_householder_apply(values, hh_vector)
    values = values * balance.float()
    if pre is not None:
        values = values.index_select(-1, pre.long())
    return _hadamard64(values).index_select(-1, local)


def _combined_root_cost(values, factor_state, groups, compact, diagonal, vector, prepared, gram):
    out = _allocate_outputs(len(values) * groups)
    _linear_a_write(values, out, 0, compact, diagonal, vector,
                    prepared=prepared, root_factor=factor_state.float())
    params = _format_outputs(out, (len(values), groups))
    residual = (values - _reconstruct_hif4(params, values.shape)).reshape(-1, groups, 64)
    return (residual * torch.einsum('ngi,gij->ngj', residual, gram)).sum((0, 2)).double()


def _combined_root_policy(calibrated, shape, samples):
    state = calibrated['activation_state']
    if len(state) == 8:
        balance_code, pre, packed_local, diagonal_code, rank_code, rank_scale, stored_vector, metadata = state
        balance = _goal_log_unpack(balance_code, metadata[:2])
        importance = _goal_log_unpack(diagonal_code, metadata[2:])
        hh_vector = _goal_householder_normalize(stored_vector)
    elif len(state) == 6:
        balance, pre, packed_local, importance, rank_code, rank_scale = state
        hh_vector = None
    else:
        raise ValueError('Unexpected core Linear state')
    width = shape[-1]
    groups = width // 64
    if not samples or importance is None:
        return torch.ones(groups, dtype=torch.float16)
    quantized = _reconstruct_hif4(calibrated['weight_params'], shape)
    blocks = quantized.reshape(-1, groups, 64)
    gram = torch.einsum('ngi,ngj->gij', blocks, blocks) / max(1, len(quantized))
    diagonal = importance.float()
    compact = diagonal.reshape(-1,64).clone()
    compact.div_(compact.mean(-1, keepdim=True).clamp_min(1e-12))
    vector = None if rank_code is None else rank_code.float() * rank_scale
    prepared = _speed_prepare_rank(diagonal, vector) if vector is not None else None
    local = _finals_linear_permutation(_linear_unpack_local(packed_local))
    # Baseline first: exact ties and empty statistics preserve factor=1.
    factors = (1.0, 0.75, 0.85, 0.925, 1.075, 1.15)
    scores = torch.zeros((len(factors), groups), dtype=torch.float64)
    for pair in samples:
        values = _combined_policy_values(pair, width, balance, pre, local, hh_vector)
        if values is None:
            continue
        baseline_cost = None
        for fi, factor in enumerate(factors):
            factor_state = torch.full((groups,), factor, dtype=torch.float16)
            cost = _combined_root_cost(values, factor_state, groups, compact, diagonal, vector, prepared, gram)
            if baseline_cost is None:
                baseline_cost = cost.sum().clamp_min(1e-24)
            scores[fi].add_(cost / baseline_cost)
    best = scores.argmin(0)
    assembled = torch.tensor(factors, dtype=torch.float16)[best].contiguous()
    control = torch.ones(groups, dtype=torch.float16)
    # Per-block choices can compete for the one full-row rank repair. Re-run
    # the assembled vector through the complete writer before freezing it.
    control_score = torch.zeros((), dtype=torch.float64)
    assembled_score = torch.zeros((), dtype=torch.float64)
    for pair in samples:
        values = _combined_policy_values(pair, width, balance, pre, local, hh_vector)
        if values is None:
            continue
        base_cost = _combined_root_cost(values, control, groups, compact, diagonal, vector, prepared, gram)
        new_cost = _combined_root_cost(values, assembled, groups, compact, diagonal, vector, prepared, gram)
        denominator = base_cost.sum().clamp_min(1e-24)
        control_score.add_(base_cost.sum() / denominator)
        assembled_score.add_(new_cost.sum() / denominator)
    return assembled if assembled_score < control_score else control


# ============================================================================
# Linear quantization: calibration writer and online shared path
# ============================================================================


def _finals_linear_quality_tree(absolute, maximum, weighted, mode='threshold104'):
    center = _engineering_nearest(maximum * (1.0 / 7.0))
    if mode == 'threshold104':
        shift = maximum > 7.0 * 1.04 * _E6M2[center]
        center = (center + shift.to(torch.int64)).clamp_max_(_E6M2.numel() - 1)
        root = _E6M2[center]
        _, lv2, lv3, mant = _fixed_scale_tree_normalized(absolute, root, return_params=True, importance=weighted)
        return (root, lv2, lv3, mant)
    if mode != 'positive3':
        raise ValueError('unsupported Linear root strategy')
    best_error = None
    for offset in (0, 1, 2):
        root = _E6M2[(center + offset).clamp_max(_E6M2.numel() - 1)]
        error_1 = _mantissa_error(absolute, root, 1.0, weighted)
        error_2 = _mantissa_error(absolute, root, 2.0, weighted)
        error_4 = _mantissa_error(absolute, root, 4.0, weighted)
        cost_low = torch.minimum(error_1, error_2).sum(dim=-1)
        cost_high = torch.minimum(error_2, error_4).sum(dim=-1)
        lv2 = cost_high < cost_low
        error = torch.where(lv2, cost_high, cost_low).sum(dim=-1)
        lv3 = torch.where(lv2[..., None], error_4 < error_2, error_2 < error_1)
        if best_error is None:
            best_error, best_root, best_lv2, best_lv3 = (error, root, lv2, lv3)
        else:
            improve = error < best_error
            best_error = torch.where(improve, error, best_error)
            best_root = torch.where(improve, root, best_root)
            best_lv2 = torch.where(improve[:, None], lv2, best_lv2)
            best_lv3 = torch.where(improve[:, None, None], lv3, best_lv3)
    multiplier = (1 + best_lv2.to(torch.int8))[..., None] * (1 + best_lv3.to(torch.int8))
    leaf_scale = best_root[:, None, None, None] * multiplier[..., None]
    mant = torch.round(absolute * (4.0 / leaf_scale)).clamp_(0.0, 7.0)
    return (best_root, best_lv2, best_lv3, mant)


def _engineering_write(values, outputs, start, importance=None):
    flat = values.reshape(-1, 64)
    for a in range(0, len(flat), _ENGINEERING_GROUPS):
        b = min(a + _ENGINEERING_GROUPS, len(flat))
        current = flat[a:b].float()
        absolute = current.abs().reshape(-1, 8, 2, 4)
        maximum = absolute.amax(dim=(-1, -2, -3))
        if not torch.isfinite(maximum).all():
            raise ValueError('HiF4 input contains non-finite values')
        weighted = None
        if importance is not None:
            idx = torch.arange(a, b).remainder(len(importance))
            weighted = importance.index_select(0, idx).reshape(-1, 8, 2, 4)
        if importance is not None:
            root, lv2, lv3, mant = _finals_linear_quality_tree(absolute, maximum, weighted, mode=_CAMPAIGN_A_MODE)
        else:
            root = _E6M2[_engineering_nearest(maximum * (1.0 / 7.0))]
            _, lv2, lv3, mant = _fixed_scale_tree_normalized(absolute, root, return_params=True)
        sign = current.sign().reshape(-1, 8, 2, 4)
        sign.masked_fill_(mant == 0, 0)
        params = (root[:, None, None, None], (1 + lv2.to(torch.int8))[:, :, None, None], (1 + lv3.to(torch.int8))[:, :, :, None], sign, mant * 0.25)
        _copy_chunk(outputs, start + a, start + b, params)


def _speed_unpack_permutation(packed):
    data = packed.to(torch.int64).reshape(-1, 3) & 255
    word = data[:, 0] | data[:, 1] << 8 | data[:, 2] << 16
    local = torch.stack((word & 63, word >> 6 & 63, word >> 12 & 63, word >> 18 & 63), dim=-1)
    local = local.reshape(-1, 64)
    return (local + torch.arange(len(local), dtype=torch.int64)[:, None] * 64).reshape(-1)


def _speed_prepare_rank(diagonal, vector):
    indices = (vector.square() / diagonal).topk(min(_LINEAR_RANK_STATIC_K, len(vector)), sorted=False).indices
    small_vector = vector.index_select(0, indices)
    small_diagonal = diagonal.index_select(0, indices)
    residual = (small_diagonal - small_vector.square()).clamp_min(1e-09)
    return indices, small_vector, small_diagonal, residual, indices // 4


def _speed_leaf_high(low, importance):
    first = torch.round(low).clamp_(0.0, 7.0)
    second = torch.round(low * 0.5).clamp_(0.0, 7.0)
    delta = second.mul_(2.0).sub_(first)
    # (q2-x)^2 - (q1-x)^2 = (q2-q1) [2(q1-x)+(q2-q1)].
    difference = first.sub_(low).mul_(2.0).add_(delta).mul_(delta)
    if importance is not None:
        difference.mul_(importance)
    return difference.sum(-1) < 0


def _speed_rank_refine(values, code, step, vector, prepared):
    indices, small_vector, small_diagonal, residual, leaf_indices = prepared
    error = values - (code * step).reshape(values.shape)
    projection = torch.matmul(error, vector).unsqueeze(-1)
    small_error = error.index_select(-1, indices)
    gradient = small_error * residual + projection * small_vector
    current = code.reshape(values.shape).index_select(-1, indices)
    small_step = step.reshape(len(values), -1).index_select(-1, leaf_indices)
    proposed = torch.round(current + gradient / (small_diagonal * small_step)).clamp_(-7, 7)
    delta = (proposed - current) * small_step
    cost = small_diagonal * delta.square() - 2 * gradient * delta
    loss, which = cost.min(-1, keepdim=True)
    chosen = torch.where(loss < 0, proposed.gather(-1, which), current.gather(-1, which))
    return indices[which], chosen


def _linear_a_write(values, outputs, start, importance, diagonal=None, vector=None, prepared=None, root_factor=None):
    rows, width = values.shape
    groups = width // 64
    if groups > _LINEAR_A_GROUPS:
        return _engineering_write(values, outputs, start, importance)
    absolute = values.abs().reshape(rows, groups, 8, 2, 4)
    maximum = absolute.amax(dim=(-1, -2, -3))
    if not torch.isfinite(maximum).all():
        raise ValueError('HiF4 input contains non-finite values')
    if root_factor is not None:
        maximum = maximum * root_factor
    center = _engineering_nearest(maximum * (1.0 / 7.0))
    if importance is not None:
        shift = maximum > 7.0 * 1.04 * _E6M2[center]
        center = (center + shift.to(torch.int64)).clamp_max_(_E6M2.numel() - 1)
    root = _E6M2[center]
    t = absolute.mul_((4.0 / root)[..., None, None, None])
    weighted = None if importance is None else importance.reshape(1, groups, 8, 2, 4)
    leafmax = t.amax(-1)
    lv2 = leafmax.amax(-1) > 14.0 * _LINEAR_A_PARENT_FACTOR
    low = t / (1 + lv2.to(torch.int8))[..., None, None]
    lv3 = _speed_leaf_high(low, weighted)
    multiplier = (1 + lv2.to(torch.int8))[..., None] * (1 + lv3.to(torch.int8))
    mant = t.mul_(multiplier[..., None].float().reciprocal()).round_().clamp_(0.0, 7.0)
    sign = values.sign().reshape_as(mant)
    sign.masked_fill_(mant == 0, 0)
    if vector is not None:
        leaf_step = root[..., None, None, None] * multiplier[..., None] * 0.25
        if prepared is None:
            prepared = _speed_prepare_rank(diagonal, vector)
        index, chosen = _speed_rank_refine(values, sign * mant, leaf_step, vector, prepared)
        sign.reshape(values.shape).scatter_(-1, index, chosen.sign())
        mant.reshape(values.shape).scatter_(-1, index, chosen.abs())
    count = rows * groups
    params = (root.reshape(count, 1, 1, 1), (1 + lv2.to(torch.int8)).reshape(count, 8, 1, 1), (1 + lv3.to(torch.int8)).reshape(count, 8, 2, 1), sign.reshape(count, 8, 2, 4), mant.mul_(0.25).reshape(count, 8, 2, 4))
    _copy_chunk(outputs, start, start + count, params)


def _hotpath_leaf_high(low, importance):
    first = torch.round(low).clamp_(0.0, 7.0)
    second = torch.round(low * 0.5).clamp_(0.0, 7.0)
    # Negate both old factors, folding each exact multiply-by-two
    # into sub(alpha=2). first/second are private rounded buffers.
    torch.sub(first, second, alpha=2, out=second)
    first.sub_(low)
    torch.sub(second, first, alpha=2, out=first)
    difference = first.mul_(second)
    if importance is not None:
        difference.mul_(importance)
    return (difference[..., 0] + difference[..., 1]).add_(difference[..., 2]).add_(difference[..., 3]) < 0


def _hotpath_rank_refine(values, code, step, vector, prepared):
    # The signed-code temporary is owned by the online writer, never a state/input.
    indices, small_vector, small_diagonal, residual, leaf_indices = prepared
    current = code.reshape(values.shape).index_select(-1, indices)
    code.mul_(step)
    error = code.reshape(values.shape)
    torch.sub(values, error, out=error)
    projection = torch.matmul(error, vector).unsqueeze(-1)
    small_error = error.index_select(-1, indices)
    gradient = small_error.mul_(residual).add_(projection * small_vector)
    small_step = step.reshape(len(values), -1).index_select(-1, leaf_indices)
    proposed = torch.round(current + gradient / (small_diagonal * small_step)).clamp_(-7, 7)
    delta = (proposed - current) * small_step
    cost = small_diagonal * delta.square() - 2 * gradient * delta
    loss, which = cost.min(-1, keepdim=True)
    chosen = torch.where(loss < 0, proposed.gather(-1, which), current.gather(-1, which))
    return indices[which], chosen


def _hotpath_linear_a_write(values, outputs, start, importance, diagonal=None, vector=None, prepared=None, root_factor=None):
    rows, width = values.shape
    groups = width // 64
    if groups > _LINEAR_A_GROUPS:
        return _engineering_write(values, outputs, start, importance)
    absolute = values.abs().reshape(rows, groups, 8, 2, 4)
    parent_max_abs = absolute.amax(dim=(-1, -2))
    maximum = parent_max_abs.amax(dim=-1)
    if not torch.isfinite(maximum).all():
        raise ValueError('HiF4 input contains non-finite values')
    if root_factor is not None:
        maximum = maximum * root_factor
    center = _engineering_nearest(maximum * (1.0 / 7.0))
    if importance is not None:
        shift = maximum > 7.0 * 1.04 * _E6M2[center]
        center = (center + shift.to(torch.int64)).clamp_max_(_E6M2.numel() - 1)
    root = _E6M2[center]
    root_scale = 4.0 / root
    weighted = None if importance is None else importance.reshape(1, groups, 8, 2, 4)
    lv2 = parent_max_abs * root_scale[..., None] > 14.0 * _LINEAR_A_PARENT_FACTOR
    parent_code = 1 + lv2.to(torch.int8)
    low = absolute.mul_((root_scale[..., None] / parent_code)[..., None, None])
    lv3 = _hotpath_leaf_high(low, weighted)
    leaf_code = 1 + lv3.to(torch.int8)
    multiplier = parent_code[..., None] * leaf_code
    mant = low.mul_(leaf_code[..., None].float().reciprocal()).round_().clamp_(0.0, 7.0)
    signed_code = torch.copysign(mant, values.reshape_as(mant))
    sign = signed_code.sign()
    if vector is not None:
        leaf_step = root[..., None, None, None] * multiplier[..., None] * 0.25
        if prepared is None:
            prepared = _speed_prepare_rank(diagonal, vector)
        index, chosen = _hotpath_rank_refine(values, signed_code, leaf_step, vector, prepared)
        sign.reshape(values.shape).scatter_(-1, index, chosen.sign())
        mant.reshape(values.shape).scatter_(-1, index, chosen.abs())
    count = rows * groups
    params = (root.reshape(count, 1, 1, 1), parent_code.reshape(count, 8, 1, 1), leaf_code.reshape(count, 8, 2, 1), sign.reshape(count, 8, 2, 4), mant.mul_(0.25).reshape(count, 8, 2, 4))
    _copy_chunk(outputs, start, start + count, params)


def _hotpath_householder_apply(values, vector):
    # The sole online caller supplies its private, newly decoded FP32 matrix.
    # Other ranks retain the original sequential arithmetic and in-place writes.
    if vector.ndim != 2 or vector.shape[0] != 2:
        for direction in vector:
            values.sub_(2.0 * (values @ direction).unsqueeze(-1) * direction)
        return values
    projection = values @ vector.transpose(0, 1)
    first, second = projection.unbind(dim=-1)
    cross = torch.dot(vector[0], vector[1])
    coefficients = torch.stack((-2.0 * first, -2.0 * second + (4.0 * cross) * first), dim=-1)
    # Both right-hand operands are independent of values before this update.
    return values.addmm_(coefficients, vector)


def _linear_dynamic_from_decoded_state(
        activation_quant, activation_scale, width, balance, pre, local,
        importance, rank_code, rank_scale, hh_vector, root_factor):
    local = _speed_unpack_permutation(local)
    diagonal = None if importance is None else importance.float()
    vector = None if rank_code is None else rank_code.float() * rank_scale
    if balance.numel() != width or width <= 0 or width % 64:
        raise ValueError('Linear activation state shape mismatch')
    balance = balance.float()
    pre = None if pre is None else pre.to(torch.int64)
    compact = None
    if importance is not None:
        compact = importance.float().reshape(-1, 64).clamp_min(1e-12)
        compact.div_(compact.mean(-1, keepdim=True))
    prepared = _speed_prepare_rank(diagonal, vector) if vector is not None else None
    quant = activation_quant.reshape(-1, width)
    scale = activation_scale.reshape(-1, width // 16)
    groups = width // 64
    outputs = _allocate_outputs(len(quant) * groups)
    row_limit = max(1, 16384 // groups)
    rows = 1 << (row_limit.bit_length() - 1)
    for start in range(0, len(quant), rows):
        end = min(start + rows, len(quant))
        if hh_vector is None:
            values = dequantize_nvfp4(quant[start:end], scale[start:end]).float().mul_(balance)
        else:
            values = _hotpath_householder_apply(dequantize_nvfp4(quant[start:end],scale[start:end]).float(),hh_vector).mul_(balance)
        if pre is not None:
            values = values.index_select(-1, pre)
        values = _hadamard64(values).index_select(-1, local)
        _hotpath_linear_a_write(values, outputs, start * groups, compact, diagonal, vector, prepared=prepared, root_factor=root_factor)
    return _format_outputs(outputs, tuple(activation_quant.shape[:-1]) + (groups,))


@torch.inference_mode()
def _goal_original_dynamic(activation_quant, activation_scale, activation_state, root_factor=None):
    width = int(activation_quant.shape[-1])
    balance, pre, local, importance, rank_code, rank_scale = activation_state
    return _linear_dynamic_from_decoded_state(
        activation_quant, activation_scale, width, balance, pre, local,
        importance, rank_code, rank_scale, None, root_factor)


@torch.inference_mode()
def _goal_householder_dynamic(activation_quant, activation_scale, activation_state, root_factor=None):
    width = int(activation_quant.shape[-1])
    balance_code, pre, local, diagonal_code, rank_code, rank_scale, stored_vector, metadata = activation_state
    balance = _goal_log_unpack(balance_code,metadata[:2])
    importance = _goal_log_unpack(diagonal_code,metadata[2:])
    hh_vector = _goal_householder_normalize(stored_vector)
    return _linear_dynamic_from_decoded_state(
        activation_quant, activation_scale, width, balance, pre, local,
        importance, rank_code, rank_scale, hh_vector, root_factor)


# ============================================================================
# Attention calibration: balance and low-rank Fisher metric
# ============================================================================


def _final_attention_hadamard(values, head_dim):
    if head_dim == 256:
        return torch.matmul(values.reshape(-1, 256), _HADAMARD256).reshape_as(values)
    if head_dim % 64 == 0:
        return _hadamard64(values)
    return values


@torch.inference_mode()
def _attn_base_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    if not calib_qkv_list or q_num_heads % kv_num_heads:
        raise ValueError('invalid Attention calibration/head configuration')
    repeats = q_num_heads // kv_num_heads
    qsum = torch.zeros((kv_num_heads, head_dim), dtype=torch.float64)
    ksum = torch.zeros_like(qsum)
    count = 0
    for sample in calib_qkv_list:
        q = dequantize_nvfp4(*sample['q']).float()
        k = dequantize_nvfp4(*sample['k']).float()
        k.sub_(k.mean(dim=0, keepdim=True))
        n = int(q.shape[0])
        qsum.add_(q.reshape(n, kv_num_heads, repeats, head_dim).square().sum(dim=(0, 2), dtype=torch.float64) / n)
        ksum.add_(k.reshape(n, kv_num_heads, head_dim).square().sum(dim=0, dtype=torch.float64) / n)
        count += 1
    balance = _safe_balance((ksum / count).sqrt().float(), (qsum / (count * repeats)).sqrt().float(), 0.5)
    qs, ks = ({'b': balance.contiguous()}, {'b': balance.clone().contiguous()})
    vs = {}
    return {'q_state': qs, 'k_state': ks, 'v_state': vs}


def _attn_lowrank_covariance(calib_qkv_list, balance, q_num_heads, kv_num_heads, head_dim):
    balance = balance.float()
    repeats = q_num_heads // kv_num_heads
    covariances = {r: torch.zeros(kv_num_heads, head_dim, head_dim, dtype=torch.float64) for r in ('q', 'k')}
    for sample in calib_qkv_list:
        q = dequantize_nvfp4(*sample['q']).float() * balance.repeat_interleave(repeats, 0).reshape(-1)
        k = dequantize_nvfp4(*sample['k']).float()
        k = (k - k.mean(0, keepdim=True)) / balance.reshape(-1)
        q = _final_attention_hadamard(q, head_dim).reshape(-1, q_num_heads, head_dim).transpose(0, 1)
        k = _final_attention_hadamard(k, head_dim).reshape(-1, kv_num_heads, head_dim).transpose(0, 1)
        k = k.repeat_interleave(repeats, 0)
        n = q.shape[1]
        kd = k.double()
        key_mass = torch.zeros(q_num_heads, n, dtype=torch.float64)
        mean_gram = torch.zeros(q_num_heads, head_dim, head_dim, dtype=torch.float64)
        k_metric = torch.zeros_like(mean_gram)
        for causal in (True, False):
            for start in range(0, n, 128):
                end = min(n, start + 128)
                query = q[:, start:end]
                logits = (query @ k.transpose(-1, -2)) * head_dim ** -0.5
                if causal:
                    mask = torch.arange(n)[None] > torch.arange(start, end)[:, None]
                    logits.masked_fill_(mask[None], float('-inf'))
                probability = logits.double().softmax(-1)
                key_mass.add_(probability.sum(1))
                key_mean = probability @ kd
                mean_gram.add_(key_mean.transpose(-1, -2) @ key_mean)
                weight = (probability * (1 - probability)).sum(-1)
                k_metric.add_(query.double().transpose(-1, -2) @ (query.double() * weight[..., None]))
        q_metric = kd.transpose(-1, -2) @ (kd * key_mass[..., None]) - mean_gram
        for role, metric in (('q', q_metric), ('k', k_metric)):
            reduced = metric.reshape(kv_num_heads, repeats, head_dim, head_dim).sum(1)
            covariances[role].add_(reduced)
    return {r: ((g + g.transpose(-1, -2)) * .5).float() for r, g in covariances.items()}


def _attn_lowrank_fit(gram, rank):
    gram = gram / gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)[:, None, None]
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues[:, -rank:].flip(-1).clamp_min(0)
    vectors = eigenvectors[:, :, -rank:].flip(-1)
    # Fix the eigenvector sign convention; signs do not affect U U^T.
    pivots = vectors.abs().argmax(dim=1, keepdim=True)
    signs = vectors.gather(1, pivots).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    vectors = vectors * signs * eigenvalues.sqrt()[:, None, :]
    scale = vectors.abs().amax(1).clamp_min(1e-12) / 127
    code = torch.round(vectors / scale[:, None, :]).clamp(-127, 127).to(torch.int8)
    diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp(1e-4, 65504).half()
    decoded = code.float() * scale[:, None, :]
    strength = decoded.square().sum(-1)
    residual = (diagonal.float() - strength).clamp_min(1e-9)
    effective = residual + strength
    indices = (strength / effective).argsort(dim=-1, descending=True, stable=True)[:, :16].to(torch.int16)
    return {'d': diagonal, 'u': code, 's': scale, 'i': indices}


def _sensitivity_balance_statistics(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    repeats = q_num_heads // kv_num_heads
    all_a, all_b, all_c, all_d = [], [], [], []
    for sample in calib_qkv_list:
        query = dequantize_nvfp4(*sample['q']).double().reshape(-1, q_num_heads, head_dim).transpose(0, 1)
        key = dequantize_nvfp4(*sample['k']).double().reshape(-1, kv_num_heads, head_dim).transpose(0, 1)
        key = key - key.mean(dim=1, keepdim=True)
        value = dequantize_nvfp4(*sample['v']).double().reshape(-1, kv_num_heads, head_dim).transpose(0, 1)
        n = int(query.shape[1])
        sampled = min(16, n)
        positions = torch.arange(sampled, dtype=torch.int64) * n // sampled
        key_index = torch.arange(n, dtype=torch.int64)
        # Q is not centered. K is centered separately within each sequence.
        a = query.square().mean(dim=1).reshape(kv_num_heads, repeats, head_dim)
        b = key.square().mean(dim=1)
        c = torch.zeros(kv_num_heads, repeats, head_dim, dtype=torch.float64)
        d = torch.zeros(kv_num_heads, head_dim, dtype=torch.float64)
        for kv_head in range(kv_num_heads):
            current_key = key[kv_head]
            current_value = value[kv_head] - value[kv_head].mean(dim=0, keepdim=True)
            for repeat in range(repeats):
                current_query = query[kv_head * repeats + repeat]
                for causal in (True, False):
                    for start in range(0, sampled, 4):
                        chosen = positions[start:start + 4]
                        block = current_query.index_select(0, chosen)
                        logits = (block @ current_key.transpose(0, 1)) * head_dim ** -0.5
                        if causal:
                            logits.masked_fill_(key_index[None] > chosen[:, None], float('-inf'))
                        probability = logits.softmax(-1)
                        key_mean = probability @ current_key
                        key_jacobian = probability[..., None] * (current_key[None] - key_mean[:, None])
                        cross = current_value.transpose(0, 1)[None] @ key_jacobian
                        c[kv_head, repeat].add_(cross.square().sum(dim=(0, 1)))
                        del key_jacobian, cross
                        # Calibration weighted V mean appears only inside the
                        # analytic Jacobian, never a fitted output-error target.
                        value_mean = probability @ current_value
                        deviation = current_value[None] - value_mean[:, None]
                        sensitivity = (probability.square() * deviation.square().sum(-1)).sum(-1)
                        d[kv_head].add_((block.square() * sensitivity[:, None]).sum(dim=0))
        # A common query/mode average gives each sequence equal weight. Keep
        # Q heads separate; K sums their effects without dividing by repeats.
        c.div_(2 * sampled)
        d.div_(2 * sampled)
        all_a.append(a)
        all_b.append(b)
        all_c.append(c)
        all_d.append(d)
    return tuple(torch.stack(items) for items in (all_a, all_b, all_c, all_d))


def _sensitivity_balance_moments(a, b, c, d, x):
    expanded = x[None, :, :, None, :]
    aq = (a * expanded).sum(-1)
    cq = (c / expanded).sum(-1)
    bk = (b / x[None]).sum(-1)
    dk = (d * x[None]).sum(-1)
    return aq, cq, bk, dk


def _sensitivity_balance_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim, balance):
    if head_dim % 64:
        return balance
    width = 256 if head_dim == 256 else 64
    blocks = head_dim // width
    repeats = q_num_heads // kv_num_heads
    a, b, c, d = _sensitivity_balance_statistics(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    samples = len(a)
    a = a.reshape(samples, kv_num_heads, repeats, blocks, width).permute(0, 1, 3, 2, 4)
    c = c.reshape(samples, kv_num_heads, repeats, blocks, width).permute(0, 1, 3, 2, 4)
    b = b.reshape(samples, kv_num_heads, blocks, width)
    d = d.reshape(samples, kv_num_heads, blocks, width)
    original = balance.double().reshape(kv_num_heads, blocks, width)
    x = original.square()
    aq, cq, bk, dk = _sensitivity_balance_moments(a, b, c, d, x)
    original_cost = (aq * cq).sum(dim=(0, 3)) + (bk * dk).sum(dim=0)
    target_mean_log = x.log().mean(dim=-1, keepdim=True)
    lower_log = 2.0 * math.log(_BALANCE_MIN)
    upper_log = 2.0 * math.log(_BALANCE_MAX)
    for sweep in range(6):
        for coordinate in range(width):
            current = x[..., coordinate]
            inverse = current.reciprocal()
            ai, ci = a[..., coordinate], c[..., coordinate]
            bi, di = b[..., coordinate], d[..., coordinate]
            aq_other = (aq - ai * current[None, :, :, None]).clamp_min(0)
            cq_other = (cq - ci * inverse[None, :, :, None]).clamp_min(0)
            bk_other = (bk - bi * inverse[None]).clamp_min(0)
            dk_other = (dk - di * current[None]).clamp_min(0)
            numerator = (ci * aq_other).sum(dim=(0, 3)) + (bi * dk_other).sum(dim=0)
            denominator = (ai * cq_other).sum(dim=(0, 3)) + (di * bk_other).sum(dim=0)
            valid = torch.isfinite(numerator) & torch.isfinite(denominator) & (numerator > 0) & (denominator > 0)
            optimum_log = 0.5 * (numerator.clamp_min(1e-300).log() - denominator.clamp_min(1e-300).log())
            proposed = (0.5 * current.log() + 0.5 * optimum_log).clamp(lower_log, upper_log).exp()
            proposed = torch.where(valid, proposed, current)
            change = proposed - current
            inverse_change = proposed.reciprocal() - inverse
            aq.add_(ai * change[None, :, :, None])
            cq.add_(ci * inverse_change[None, :, :, None])
            bk.add_(bi * inverse_change[None])
            dk.add_(di * change[None])
            x[..., coordinate].copy_(proposed)
        # Fix the homogeneous block scale. The scalar bisection projects logs
        # onto the original geometric mean and original box bounds together.
        log_x = x.log()
        left = lower_log - log_x.amax(dim=-1, keepdim=True)
        right = upper_log - log_x.amin(dim=-1, keepdim=True)
        for projection_step in range(32):
            middle = (left + right) * 0.5
            mean_log = (log_x + middle).clamp(lower_log, upper_log).mean(dim=-1, keepdim=True)
            below = mean_log < target_mean_log
            left = torch.where(below, middle, left)
            right = torch.where(below, right, middle)
        x = (log_x + (left + right) * 0.5).clamp(lower_log, upper_log).exp()
        aq, cq, bk, dk = _sensitivity_balance_moments(a, b, c, d, x)
    proposed_balance = x.sqrt().to(balance.dtype).clamp(_BALANCE_MIN, _BALANCE_MAX)
    stored_x = proposed_balance.double().square()
    aq, cq, bk, dk = _sensitivity_balance_moments(a, b, c, d, stored_x)
    proposed_cost = (aq * cq).sum(dim=(0, 3)) + (bk * dk).sum(dim=0)
    accept = torch.isfinite(original_cost) & torch.isfinite(proposed_cost) & (proposed_cost < original_cost)
    return torch.where(accept[..., None], proposed_balance, balance.reshape(kv_num_heads, blocks, width)).reshape_as(balance).contiguous()


# ============================================================================
# Attention online quantization: Q/K repair and V local correction
# ============================================================================


def _final_attention_output(tokens, channels):
    return _format_outputs(_allocate_outputs(tokens * (channels // 64)), (tokens, channels // 64))


def _joint_attn_prepare_metric(state, heads):
    # Callback-local preprocessing only; no state mutation or cross-call cache.
    if 'd' not in state:
        return None
    repeats = heads // len(state['d'])
    diagonal = state['d'].float()
    vectors = state['u'].float() * state['s'][:, None, :]
    indices = state['i'].long()
    if repeats != 1:
        diagonal = diagonal.repeat_interleave(repeats, 0)
        vectors = vectors.repeat_interleave(repeats, 0)
        indices = indices.repeat_interleave(repeats, 0)
    strength = vectors.square().sum(-1)
    residual = (diagonal - strength).clamp_min(1e-9)
    effective = residual + strength
    small_vectors = vectors.gather(1, indices[:, :, None].expand(-1, -1, vectors.shape[-1]))
    small_residual = residual.gather(-1, indices)
    small_effective = effective.gather(-1, indices)
    selected_gram = small_vectors @ small_vectors.transpose(-1, -2)
    selected_gram.diagonal(dim1=-2, dim2=-1).copy_(small_effective)
    leaf_indices = indices // 4
    return residual, effective, vectors, indices, small_vectors, small_residual, small_effective, selected_gram, leaf_indices


def _speed_attn_sum4(values):
    first, second, third, fourth = values.unbind(dim=-1)
    return (first + second).add_(third).add_(fourth)


def _speed_attn_sum2(values):
    first, second = values.unbind(dim=-1)
    return first + second


def _speed_attn_fixed_scale_tree_normalized(
        absolute, group_scale, *, return_params, importance=None,
        physical_cost=True, restore_physical_mantissa=False):
    scaled = absolute * (4.0 / group_scale)[:, None, None, None]
    leaf_maximum = scaled.amax(dim=-1)
    lv2 = leaf_maximum.amax(dim=-1) > (14.0 * 1.1)
    parent = 1 + lv2.to(torch.int8)
    lv3 = leaf_maximum > 7.0 * parent[..., None] * 1.1
    multiplier = parent[..., None] * (1 + lv3.to(torch.int8))
    if restore_physical_mantissa:
        leaf_scale = group_scale[:, None, None, None] * multiplier[..., None]
        mantissa = torch.round(
            absolute * (4.0 / leaf_scale)).clamp_(0.0, 7.0)
    else:
        mantissa = torch.round(
            scaled * multiplier[..., None].float().reciprocal()).clamp_(0.0, 7.0)
    return (None, lv2, lv3, mantissa)


def _speed_attn_engineering_write(values, outputs, start, importance=None):
    flat = values.reshape(-1, 64)
    for a in range(0, len(flat), _ENGINEERING_GROUPS):
        b = min(a + _ENGINEERING_GROUPS, len(flat))
        current = flat[a:b].float()
        absolute = current.abs().reshape(-1, 8, 2, 4)
        maximum = absolute.amax(dim=(-1, -2, -3))
        if not torch.isfinite(maximum).all():
            raise ValueError('HiF4 input contains non-finite values')
        weighted = None
        if importance is not None:
            idx = torch.arange(a, b).remainder(len(importance))
            weighted = importance.index_select(0, idx).reshape(-1, 8, 2, 4)
        if importance is not None:
            root, lv2, lv3, mant = _finals_linear_quality_tree(absolute, maximum, weighted, mode=_CAMPAIGN_A_MODE)
        else:
            root = _E6M2[_engineering_nearest(maximum * (1.0 / 7.0))]
            _, lv2, lv3, mant = _speed_attn_fixed_scale_tree_normalized(absolute, root, return_params=True)
        sign = current.sign().reshape(-1, 8, 2, 4)
        sign.masked_fill_(mant == 0, 0)
        params = (root[:, None, None, None], (1 + lv2.to(torch.int8))[:, :, None, None], (1 + lv3.to(torch.int8))[:, :, :, None], sign, mant * 0.25)
        _copy_chunk(outputs, start + a, start + b, params)


def _attn_fused_write(values, outputs, start, heads, dimension, metric, steps):
    if metric is None:
        _speed_attn_engineering_write(values, outputs, start)
        return
    flat = values.reshape(-1, 64)
    leaf_absolute_maximum = flat.abs().reshape(-1, 8, 2, 4).amax(dim=-1)
    maximum = leaf_absolute_maximum.amax(dim=(-1, -2))
    if not torch.isfinite(maximum).all():
        raise ValueError('HiF4 input contains non-finite values')
    root = _E6M2[_engineering_nearest(maximum * (1.0 / 7.0))]
    reciprocal_root = 4.0 / root
    # Positive scalar multiplication commutes with the four-value maximum.
    leaf_maximum = leaf_absolute_maximum.mul_(reciprocal_root[:, None, None])
    lv2 = leaf_maximum.amax(dim=-1) > (14.0 * 1.1)
    scale2 = 1 + lv2.to(torch.int8)
    lv3 = leaf_maximum > 7.0 * scale2[..., None] * 1.1
    scale3 = 1 + lv3.to(torch.int8)
    multiplier = scale2[..., None] * scale3
    leaf = root[:, None, None, None] * scale2[:, :, None, None] * scale3[:, :, :, None] * 0.25
    # Combine the power-of-two leaf factor in the small scale tensor;
    # the full signed values need just one multiplication before rounding.
    code_scale = reciprocal_root[:, None, None] * multiplier.float().reciprocal()
    signed_code = flat.reshape(-1, 8, 2, 4) * code_scale[..., None]
    signed_code.round_().clamp_(-7.0, 7.0)
    del leaf_absolute_maximum, leaf_maximum, maximum, reciprocal_root, lv2, lv3, multiplier, code_scale
    _attn_fused_refine(values, signed_code, leaf, heads, dimension, metric, steps)
    end = start + len(flat)
    params = (root[:, None, None, None], scale2[:, :, None, None], scale3[:, :, :, None])
    _copy_chunk(outputs[:3], start, end, params)
    # Codes are exact INT8 values; capture their signs before abs_ reuse.
    outputs[3][start:end].copy_(signed_code).sign_()
    torch.mul(signed_code.abs_(), 0.25, out=outputs[4][start:end])


def _attn_fused_refine(values, signed_code, leaf, heads, dimension, metric, steps):
    n = len(values)
    residual, effective, vectors, indices, small_vectors, small_residual, small_effective, selected_gram, leaf_indices = metric
    error = (signed_code * leaf).reshape_as(values)
    torch.sub(values, error, out=error)
    error = error.reshape(n, heads, dimension)
    projection = torch.einsum('nhd,hdr->nhr', error, vectors)
    idx = indices[None].expand(n, -1, -1)
    small_error = error.gather(-1, idx)
    code_view = signed_code.reshape(n, heads, dimension)
    current = code_view.gather(-1, idx)
    step = leaf.reshape(n, heads, dimension // 4).gather(-1, leaf_indices[None].expand(n, -1, -1))
    gradient = small_error * small_residual + torch.einsum('nhr,hjr->nhj', projection, small_vectors)
    del error, projection, small_error
    denominator = small_effective * step
    code_weight = denominator * step
    gram_rows = selected_gram[None].expand(n, -1, -1, -1)
    for iteration in range(steps):
        normalized = gradient / denominator
        delta_code = (current + normalized).round_().clamp_(-7, 7).sub_(current)
        cost = delta_code.sub(normalized, alpha=2).mul_(delta_code).mul_(code_weight)
        loss, which = cost.min(-1, keepdim=True)
        accepted_code = torch.where(loss < 0, delta_code.gather(-1, which), 0.0)
        current.scatter_add_(-1, which, accepted_code)
        if iteration + 1 < steps:
            change = accepted_code * step.gather(-1, which)
            row_index = which[..., None].expand(-1, -1, -1, selected_gram.shape[-1])
            picked = gram_rows.gather(2, row_index).squeeze(2)
            gradient.addcmul_(change, picked, value=-1)
    code_view.scatter_(-1, idx, current)


def _joint_v_mantissa_error_inplace(absolute, group_scale, multiplier, importance=None):
    scale = group_scale[:, None, None, None] * multiplier
    code = (absolute * (4.0 / scale)).round_().clamp_(0.0, 7.0)
    squared_error = torch.addcmul(absolute, code, 0.25 * scale, value=-1, out=code).square_()
    if importance is not None:
        squared_error.mul_(importance)
    return _speed_attn_sum4(squared_error)


def _joint_v_root_codes(values):
    current = values.reshape(-1, 64).float()
    absolute = current.abs().reshape(-1, 8, 2, 4)
    maximum = absolute.amax(dim=(-1, -2, -3))
    if not torch.isfinite(maximum).all():
        raise ValueError('HiF4 input contains non-finite values')
    center = _engineering_nearest(maximum * (1.0 / 7.0))
    best_error = None
    for offset in (0, 1):
        root = _E6M2[(center + offset).clamp(0, _E6M2.numel() - 1)]
        error_1 = _joint_v_mantissa_error_inplace(absolute, root, 1.0)
        error_2 = _joint_v_mantissa_error_inplace(absolute, root, 2.0)
        error_4 = _joint_v_mantissa_error_inplace(absolute, root, 4.0)
        cost_low = _speed_attn_sum2(torch.minimum(error_1, error_2))
        cost_high = _speed_attn_sum2(torch.minimum(error_2, error_4))
        lv2 = cost_high < cost_low
        error = torch.where(lv2, cost_high, cost_low).sum(dim=-1)
        lv3 = torch.where(lv2[..., None], error_4 < error_2, error_2 < error_1)
        if best_error is None:
            best_error, best_root = (error, root)
            best_lv2, best_lv3 = (lv2, lv3)
        else:
            improve = error < best_error
            best_error = torch.where(improve, error, best_error)
            best_root = torch.where(improve, root, best_root)
            best_lv2 = torch.where(improve[:, None], lv2, best_lv2)
            best_lv3 = torch.where(improve[:, None, None], lv3, best_lv3)
    scale2 = 1 + best_lv2.to(torch.int8)
    scale3 = 1 + best_lv3.to(torch.int8)
    multiplier = scale2[..., None] * scale3
    leaf_scale = best_root[:, None, None, None] * multiplier[..., None]
    # The error search is complete; do not keep a full absolute block alive.
    del absolute
    # Symmetric integer rounding avoids materializing sign and quarter-mantissa
    # fields only to multiply them back together in the immediately next step.
    code = (current.reshape(-1, 8, 2, 4) * (4.0 / leaf_scale)).round_().clamp_(-7.0, 7.0)
    leaf = leaf_scale.mul_(0.25)
    return (best_root[:, None, None, None], scale2[:, :, None, None],
            scale3[:, :, :, None], code, leaf)


def _speed_v_write(values, output, start, end, first_group, window=64, gain=0.1):
    # The caller bounds this block to <=8192 groups and complete local windows.
    # E6M2 roots, integer scales, signs, and quarter codes are exactly BF16/int8
    # representable, so bypassing their intermediate output casts loses no bits.
    root, lv2, lv3, code, leaf = _joint_v_root_codes(values)
    shape = values.shape
    reconstruction = (code * leaf).reshape(shape)
    torch.sub(values, reconstruction, out=reconstruction)
    residual = reconstruction
    del reconstruction
    # The sole caller owns this freshly decoded FP32 chunk and never reads it
    # after this writer. Root/code/leaf and residual are already independent.
    working = values
    full_rows = (shape[0] // window) * window
    if full_rows:
        residual_windows = residual[:full_rows].reshape(-1, window, shape[1])
        working[:full_rows].reshape(-1, window, shape[1]).add_(
            residual_windows.mean(dim=1, keepdim=True), alpha=gain)
        del residual_windows
    if full_rows < shape[0]:
        mean_error = residual[full_rows:].mean(dim=0, keepdim=True)
        working[full_rows:].add_(mean_error, alpha=gain)
        del mean_error
    code_shape = code.shape
    del residual, code
    corrected_code = working.reshape(code_shape).div_(leaf).round_().clamp_(-7.0, 7.0)
    params = (root, lv2, lv3, corrected_code.sign(), corrected_code.abs_().mul_(0.25))
    groups = shape[1] // 64
    for key, value in zip(('scale_factor', 'scale_lv2', 'scale_lv3', 'sign', 'mant'), params):
        output[key][start:end, first_group:first_group + groups].copy_(
            value.reshape((shape[0], groups) + tuple(value.shape[1:])))


# ============================================================================
# The six public competition interfaces
# ============================================================================


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    calibrated = _combined_core_calibration(weight_quant, weight_scale, calib_activation_list)
    factor = _combined_root_policy(calibrated, weight_quant.shape, calib_activation_list)
    calibrated['activation_state'] = calibrated['activation_state'] + (factor,)
    return calibrated


@torch.inference_mode()
def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    root_factor = activation_state[-1].float()
    state = activation_state[:-1]
    if len(state) == 6:
        return _goal_original_dynamic(activation_quant, activation_scale, state, root_factor=root_factor)
    return _goal_householder_dynamic(activation_quant, activation_scale, state, root_factor=root_factor)


@torch.inference_mode()
def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states = _attn_base_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    if head_dim < 64 or head_dim >= 32768 or head_dim % 4:
        return states
    balance = states['k_state']['b'].half()
    balance = _sensitivity_balance_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim, balance)
    states['q_state']['b'] = balance.clone()
    states['k_state']['b'] = balance.clone()
    covariance = _attn_lowrank_covariance(calib_qkv_list, balance, q_num_heads, kv_num_heads, head_dim)
    for role in ('q', 'k'):
        states[role + '_state'].update(_attn_lowrank_fit(covariance[role], _ATTN_LR_RANK))
    states['q_state']['i'] = _attn_q_coverage_indices(
        calib_qkv_list, q_num_heads, head_dim, states['q_state'])
    return states


@torch.inference_mode()
def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    metric = _joint_attn_prepare_metric(q_state, q_num_heads)
    balance = q_state['b']
    kv_num_heads = int(balance.shape[0])
    if q_num_heads % kv_num_heads or int(balance.shape[1]) != head_dim or int(q_quant.shape[-1]) != q_num_heads * head_dim:
        raise ValueError('Q head configuration does not match calibration')
    per_head = balance.repeat_interleave(q_num_heads // kv_num_heads, dim=0).reshape(-1)
    tokens, channels = q_quant.shape
    groups = channels // 64
    output = _allocate_outputs(tokens * groups)
    rows = max(1, 32768 // groups)
    for start in range(0, tokens, rows):
        end = min(start + rows, tokens)
        values = dequantize_nvfp4(q_quant[start:end], q_scale[start:end]).float()
        values.mul_(per_head)
        values = _final_attention_hadamard(values, head_dim)
        _attn_fused_write(values, output, start * groups, q_num_heads, head_dim, metric, _ATTN_Q_STEPS)
    return _format_outputs(output, (tokens, groups))


@torch.inference_mode()
def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    metric = _joint_attn_prepare_metric(k_state, kv_num_heads)
    balance = k_state['b']
    if tuple(balance.shape) != (kv_num_heads, head_dim) or int(k_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError('K head configuration does not match calibration')
    tokens, channels = k_quant.shape
    rows = max(1, 524288 // channels)
    # Reuse the decode only while its FP32 buffer remains at most 4 MiB.
    decoded = dequantize_nvfp4(k_quant, k_scale).float() if tokens * channels <= 1048576 else None
    total = torch.zeros(channels, dtype=torch.float64)
    for start in range(0, tokens, rows):
        end = min(start + rows, tokens)
        values = decoded[start:end] if decoded is not None else dequantize_nvfp4(k_quant[start:end], k_scale[start:end]).float()
        total.add_(values.sum(dim=0, dtype=torch.float64))
    mean = (total / tokens).float()
    groups = channels // 64
    output = _allocate_outputs(tokens * groups)
    quant_rows = max(1, min(tokens, 1024)) if decoded is not None else rows
    for start in range(0, tokens, quant_rows):
        end = min(start + quant_rows, tokens)
        values = decoded[start:end] if decoded is not None else dequantize_nvfp4(k_quant[start:end], k_scale[start:end]).float()
        values.sub_(mean).div_(balance.reshape(-1))
        values = _final_attention_hadamard(values, head_dim)
        _attn_fused_write(values, output, start * groups, kv_num_heads, head_dim, metric, _ATTN_K_STEPS)
    return _format_outputs(output, (tokens, groups))


@torch.inference_mode()
def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    tokens, channels = v_quant.shape
    if channels != kv_num_heads * head_dim or tokens <= 0:
        raise ValueError('V head configuration does not match input')
    output = _final_attention_output(tokens, channels)
    window = 64
    for first_channel in range(0, channels, 2048):
        last_channel = min(channels, first_channel + 2048)
        width = last_channel - first_channel
        row_budget = max(1, 16384 * 64 // width)
        rows = max(window, (row_budget // window) * window)
        for start in range(0, tokens, rows):
            end = min(tokens, start + rows)
            values = dequantize_nvfp4(
                v_quant[start:end, first_channel:last_channel],
                v_scale[start:end, first_channel // 16:last_channel // 16]).float()
            _speed_v_write(values, output, start, end, first_channel // 64, window=window, gain=0.2)
    return output




def _attn_q_coverage_cost(records, state, heads, dimension):
    metric = _joint_attn_prepare_metric(state, heads)
    residual = metric[0].double()
    vectors = metric[2].double()
    metric_heads = len(state['d'])
    repeats = heads // metric_heads
    total = torch.zeros(metric_heads, dtype=torch.float64)
    for values, initial_code, leaf, weight in records:
        code = initial_code.clone()
        # Execute the unchanged deployed two-step repair, not a continuous
        # best-coordinate oracle or the coverage training objective.
        _attn_fused_refine(values, code, leaf, heads, dimension, metric, _ATTN_Q_STEPS)
        error = (values - (code * leaf).reshape_as(values)).reshape(-1, heads, dimension).double()
        projection = torch.einsum('nhd,hdr->nhr', error, vectors)
        loss = (error.square() * residual).sum(-1) + projection.square().sum(-1)
        total.add_(loss.reshape(len(values), metric_heads, repeats).sum((0, 2)), alpha=weight)
    return total

def _attn_q_coverage_indices(calib_qkv_list, heads, dimension, state):
    # Preserve the old path for dimensions with cross-head HiF4 root groups.
    if dimension % 64 or 'd' not in state:
        return state['i']
    metric_heads = len(state['d'])
    repeats = heads // metric_heads
    metric = _joint_attn_prepare_metric(state, heads)
    residual, effective, vectors = metric[:3]
    balance = state['b'].float().repeat_interleave(repeats, 0).reshape(-1)
    records, gain_tables = [], []
    for sample in calib_qkv_list:
        quant, scale = sample['q']
        tokens = len(quant)
        sampled = min(tokens, 16)
        if sampled == 0:
            continue
        # One midpoint per equal-width stratum; every legal sequence remains
        # represented. N/m preserves the original token-weighted policy.
        positions = ((2 * torch.arange(sampled, dtype=torch.int64) + 1) * tokens) // (2 * sampled)
        values = dequantize_nvfp4(quant.index_select(0, positions), scale.index_select(0, positions)).float()
        values.mul_(balance)
        values = _final_attention_hadamard(values, dimension)
        flat = values.reshape(-1, 64)
        absolute = flat.abs().reshape(-1, 8, 2, 4)
        maximum = absolute.amax(dim=(-1, -2, -3))
        if not torch.isfinite(maximum).all():
            return state['i']
        root = _E6M2[_engineering_nearest(maximum * (1.0 / 7.0))]
        _, lv2, lv3, mantissa = _speed_attn_fixed_scale_tree_normalized(
            absolute, root, return_params=True, physical_cost=False)
        scale2 = 1 + lv2.to(torch.int8)
        scale3 = 1 + lv3.to(torch.int8)
        leaf = root[:, None, None, None] * scale2[:, :, None, None] * scale3[:, :, :, None] * 0.25
        initial_code = flat.sign().reshape_as(mantissa).mul_(mantissa)
        error = (values - (initial_code * leaf).reshape_as(values)).reshape(sampled, heads, dimension)
        projection = torch.einsum('nhd,hdr->nhr', error, vectors)
        gradient = error * residual + torch.einsum('nhr,hdr->nhd', projection, vectors)
        step = leaf.reshape(sampled, heads, dimension // 4).repeat_interleave(4, -1)
        current = initial_code.reshape(sampled, heads, dimension)
        normalized = gradient / (effective * step)
        change = (current + normalized).round().clamp(-7, 7) - current
        cost = change * (change - 2 * normalized) * (effective * step.square())
        gain = (-cost).clamp_min(0)
        gain = torch.where(torch.isfinite(gain), gain, torch.zeros_like(gain))
        weight = tokens / sampled
        gain = gain.reshape(sampled, metric_heads, repeats, dimension).permute(1, 0, 2, 3)
        gain_tables.append(gain.reshape(metric_heads, sampled * repeats, dimension).double() * weight)
        records.append((values, initial_code, leaf, weight))
    if not records:
        return state['i']
    gain = torch.cat(gain_tables, dim=1)
    covered = torch.zeros((metric_heads, gain.shape[1], 1), dtype=torch.float64)
    selected = torch.zeros((metric_heads, dimension), dtype=torch.bool)
    # Stable original energy ordering resolves ties, including zero-gain rows.
    base_vectors = state['u'].float() * state['s'][:, None, :]
    strength = base_vectors.square().sum(-1)
    base_effective = (state['d'].float() - strength).clamp_min(1e-9) + strength
    priority = (strength / base_effective).argsort(dim=-1, descending=True, stable=True)
    indices = []
    for slot in range(state['i'].shape[-1]):
        increment = (gain - covered).clamp_min(0).sum(1)
        increment.masked_fill_(selected, float('-inf'))
        which = increment.gather(-1, priority).argmax(-1, keepdim=True)
        coordinate = priority.gather(-1, which)
        indices.append(coordinate)
        selected.scatter_(-1, coordinate, True)
        chosen = gain.gather(-1, coordinate[:, None, :].expand(-1, gain.shape[1], -1))
        covered = torch.maximum(covered, chosen)
    proposal = torch.cat(indices, dim=-1).to(torch.int16)
    proposed_state = dict(state)
    proposed_state['i'] = proposal
    original_loss = _attn_q_coverage_cost(records, state, heads, dimension)
    proposed_loss = _attn_q_coverage_cost(records, proposed_state, heads, dimension)
    accept = torch.isfinite(original_loss) & torch.isfinite(proposed_loss) & (proposed_loss < original_loss)
    return torch.where(accept[:, None], proposal, state['i']).contiguous()
