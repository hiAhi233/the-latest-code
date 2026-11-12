"""方案 §10：ResNet-101 patch 编码器的形状与冻结约定。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from medrrg.features.resnet_visual import (ResNetPatchEncoder, encoder_summary,
                                           slice_to_tensor, state_dict_sha256)


@pytest.fixture(scope="module")
def encoder():
    # pretrained=False：测试不下载权重，只验证结构与形状
    return ResNetPatchEncoder(backbone="resnet101", pretrained=False,
                              patch_grid=3, output_dim=16, frozen=True)


def test_c5_shape(encoder):
    images = torch.rand(2, 3, 64, 64)
    c5 = encoder.forward_c5(images)
    # 64 输入经过 32 倍下采样 -> 2×2
    assert tuple(c5.shape) == (2, 2048, 2, 2)


def test_patch_pooling_flattens_to_nine_patches(encoder):
    images = torch.rand(2, 3, 64, 64)
    pooled = encoder.pool_and_project(encoder.forward_c5(images), project=False)
    assert tuple(pooled.shape) == (2, 9, 2048)


def test_projection_to_output_dim(encoder):
    tokens = encoder(torch.rand(3, 3, 64, 64))
    assert tuple(tokens.shape) == (3, 9, 16)
    assert encoder.num_patches == 9


def test_frozen_encoder_has_no_gradients(encoder):
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.training is False


def test_slice_to_tensor_replicates_gray_channel():
    array = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    tensor = slice_to_tensor(array)
    assert tuple(tensor.shape) == (3, 4, 4)
    for channel in range(3):
        np.testing.assert_allclose(tensor[channel].numpy(), array, atol=1e-6)


def test_state_dict_sha_is_deterministic(encoder):
    assert state_dict_sha256(encoder.body) == state_dict_sha256(encoder.body)


def test_patch_tokens_differ_across_positions(encoder):
    tokens = encoder(torch.rand(1, 3, 64, 64))[0]
    # 不同 patch 的输入感受野不同，token 不应完全相同
    assert not torch.allclose(tokens[0], tokens[8])


def test_projection_is_deterministic_across_processes():
    """导出缓存保存投影后的 256 维 tokens，投影必须由 seed 唯一确定。

    Body 权重在 pretrained=True 时来自 ImageNet，跨进程一致；这里只验证
    2048→d 投影层：相同 seed 得到相同投影，不同 seed 得到不同投影。
    """
    first = ResNetPatchEncoder(pretrained=False, patch_grid=2, output_dim=8, proj_seed=7)
    second = ResNetPatchEncoder(pretrained=False, patch_grid=2, output_dim=8, proj_seed=7)
    other = ResNetPatchEncoder(pretrained=False, patch_grid=2, output_dim=8, proj_seed=8)
    assert encoder_summary(first)["projection_sha256"] == \
        encoder_summary(second)["projection_sha256"]
    assert encoder_summary(first)["projection_sha256"] != \
        encoder_summary(other)["projection_sha256"]
    c5 = torch.randn(2, 2048, 2, 2)
    assert torch.allclose(first.pool_and_project(c5), second.pool_and_project(c5),
                          atol=1e-6)
    assert not torch.allclose(first.pool_and_project(c5), other.pool_and_project(c5),
                              atol=1e-6)
    # 同一进程内重复前向也一致
    assert torch.allclose(first.pool_and_project(c5), first.pool_and_project(c5),
                          atol=1e-6)
