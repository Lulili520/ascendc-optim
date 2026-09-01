"""Deterministic reference adapters for kernels that replace model subgraphs.

An adapter returns values keyed by C++ extension parameter name and the exact
PyTorch result of that same fused subgraph.  It must derive everything from the
fixed model and original input; synthetic or guessed tensors are forbidden.
"""

from __future__ import annotations


def _bam(model, inputs, torch):
    x = inputs[0]
    channel_map = model.channel_attn(x).contiguous()
    spatial_map = model.spatial_attn(x).contiguous()
    expected = x + x * torch.sigmoid(channel_map + spatial_map)
    return {
        "channel_map": channel_map,
        "spatial_map": spatial_map,
    }, expected


def _block_sparse_attention(model, inputs, torch):
    x = inputs[0]
    batch, tokens, _ = x.shape
    heads, width, block = model.n_heads, model.d_k, model.block_size
    blocks = tokens // block
    def project(layer):
        return (layer(x).view(batch, tokens, heads, width).transpose(1, 2)
                .contiguous().view(batch, heads, blocks, block, width))
    q, k, v = project(model.W_q), project(model.W_k), project(model.W_v)
    scale = torch.tensor([width ** -0.5], dtype=q.dtype, device=q.device).contiguous()
    expected = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    expected = torch.matmul(expected, v)
    return {"q": q, "k": k, "v": v, "scale": scale}, expected


def _cbam(model, inputs, torch):
    x = inputs[0]
    ca = model.ca(x).contiguous()
    weighted = x * ca
    sa = model.sa(weighted).contiguous()
    return {"ca": ca, "sa": sa}, weighted * sa + x


def _parnet(model, inputs, torch):
    x = inputs[0]
    x1 = model.conv1x1(x).contiguous()
    x2 = model.conv3x3(x).contiguous()
    x3 = (model.sse(x) * x).contiguous()
    return {"x1": x1, "x2": x2, "x3": x3}, model.silu(x1 + x2 + x3)


def _residual_attention(model, inputs, torch):
    x = inputs[0]
    la = torch.tensor([model.la], dtype=x.dtype, device=x.device).contiguous()
    flat = x.flatten(2)
    expected = flat.mean(dim=2) + la * flat.max(dim=2).values
    return {"la": la}, expected


def _triplet_attention(model, inputs, torch):
    x = inputs[0]
    x_ch = model.ch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
    x_cw = model.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
    x_hw = model.hw(x).contiguous()
    return {"x_ch": x_ch, "x_cw": x_cw, "x_hw": x_hw}, (x_ch + x_cw + x_hw) / 3.0


def _axial_attention(model, inputs, torch):
    # The exported AscendC op is the attention core inside the first axial
    # branch, not the surrounding permutation/projection or the whole model.
    branch = model.axial_attentions[0]
    axial = inputs[0].permute(*branch.permutation).contiguous()
    shape = axial.shape
    axial = axial.reshape(-1, shape[-2], shape[-1])
    attention = branch.fn
    q = attention.to_q(axial)
    k, v = attention.to_kv(axial).chunk(2, dim=-1)
    batch, tokens, _ = q.shape
    heads, width = attention.heads, attention.dim_heads

    def merge_heads(value):
        return (value.reshape(batch, tokens, heads, width)
                .transpose(1, 2).reshape(batch * heads, tokens, width)
                .contiguous())

    q, k, v = map(merge_heads, (q, k, v))
    scale = width ** -0.5
    expected = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
    expected = torch.bmm(expected, v)
    return {"q": q, "k": k, "v": v, "scale_attr": scale}, expected


def _gemm(model, inputs, torch):
    return {"weight_t": model.weight.transpose(0, 1).contiguous()}, model(inputs[0])


def _gated_channel_transform(model, inputs, torch):
    return {
        "alpha": model.alpha.reshape(-1).contiguous(),
        "gamma": model.gamma.reshape(-1).contiguous(),
        "beta": model.beta.reshape(-1).contiguous(),
        "epsilon": float(model.epsilon),
    }, model(inputs[0])


def _paged_attention(model, inputs, torch):
    causal_flag = torch.tensor(
        [1 if model.causal else 0], dtype=torch.int32, device=inputs[0].device
    ).contiguous()
    return {"causal_flag": causal_flag}, model(*inputs)


def _se_attention(model, inputs, torch):
    return {
        "w1": model.fc[0].weight.contiguous(),
        "w2": model.fc[2].weight.contiguous(),
    }, model(inputs[0])


def _srm(model, inputs, torch):
    return {
        "cfc_weight": model.cfc.weight.contiguous(),
        "bn_weight": model.bn.weight.contiguous(),
        "bn_bias": model.bn.bias.contiguous(),
        "bn_running_mean": model.bn.running_mean.contiguous(),
        "bn_running_var": model.bn.running_var.contiguous(),
        "eps": float(model.bn.eps),
    }, model(inputs[0])


def _cot_attention(model, inputs, torch):
    x = inputs[0]
    op = model.cot_attention
    batch, channels, height, width = x.shape
    k1 = op.key_embed(x)
    v = op.value_embed(x).reshape(batch, channels, -1)
    att = op.attention_embed(torch.cat([k1, x], dim=1))
    att = att.reshape(batch, channels, op.kernel_size ** 2, height, width)
    att = att.mean(2).reshape(batch, channels, -1).contiguous()
    k1 = k1.contiguous()
    v = v.contiguous()
    expected = k1 + (torch.softmax(att, dim=-1) * v).reshape_as(k1)
    return {"k1": k1, "att": att, "v": v}, expected


def _coord_att(model, inputs, torch):
    x = inputs[0]
    _, _, height, width = x.shape
    x_h = model.pool_h(x)
    x_w = model.pool_w(x).permute(0, 1, 3, 2)
    y = model.act(model.bn1(model.conv1(torch.cat([x_h, x_w], dim=2))))
    x_h, x_w = torch.split(y, [height, width], dim=2)
    a_h = model.conv_h(x_h).sigmoid().expand_as(x).contiguous()
    a_w = model.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid().expand_as(x).contiguous()
    return {"a_w": a_w, "a_h": a_h}, x * a_w * a_h


def _criss_cross_attention(model, inputs, torch):
    x = inputs[0]
    batch, _, height, width = x.shape
    query = model.query_conv(x)
    query_h = query.permute(0, 3, 1, 2).contiguous().reshape(batch * width, -1, height).permute(0, 2, 1)
    query_w = query.permute(0, 2, 1, 3).contiguous().reshape(batch * height, -1, width).permute(0, 2, 1)
    key = model.key_conv(x)
    key_h = key.permute(0, 3, 1, 2).contiguous().reshape(batch * width, -1, height)
    key_w = key.permute(0, 2, 1, 3).contiguous().reshape(batch * height, -1, width)
    value = model.value_conv(x)
    value_h = value.permute(0, 3, 1, 2).contiguous().reshape(batch * width, -1, height)
    value_w = value.permute(0, 2, 1, 3).contiguous().reshape(batch * height, -1, width)
    energy_h = (torch.bmm(query_h, key_h) + model.INF(batch, height, width, x.device))
    energy_h = energy_h.reshape(batch, width, height, height).permute(0, 2, 1, 3)
    energy_w = torch.bmm(query_w, key_w).reshape(batch, height, width, width)
    attention = model.softmax(torch.cat([energy_h, energy_w], dim=3))
    att_h = attention[..., :height].permute(0, 2, 1, 3).contiguous().reshape(batch * width, height, height)
    att_w = attention[..., height:height + width].contiguous().reshape(batch * height, width, width)
    out_h = torch.bmm(value_h, att_h.permute(0, 2, 1)).reshape(batch, width, -1, height).permute(0, 2, 3, 1).contiguous()
    out_w = torch.bmm(value_w, att_w.permute(0, 2, 1)).reshape(batch, height, -1, width).permute(0, 2, 1, 3).contiguous()
    gamma = model.gamma.reshape(1).contiguous()
    return {"out_h": out_h, "out_w": out_w, "gamma": gamma}, gamma * (out_h + out_w) + x


def _da_module(model, inputs, torch):
    x = inputs[0]
    batch, channels, height, width = x.shape
    p_out = model.position_attention_module(x).contiguous()
    c_out = model.channel_attention_module(x).contiguous()
    expected = p_out.permute(0, 2, 1).reshape(batch, channels, height, width)
    expected = expected + c_out.reshape(batch, channels, height, width)
    return {"p_out": p_out, "c_out": c_out, "h": height, "w": width}, expected


def _gc_module(model, inputs, torch):
    x = inputs[0]
    y = model.transform(model.context_modeling(x)).contiguous()
    return {"y_bc11": y}, x + y


def _project_heads(model, x):
    batch, tokens, _ = x.shape
    q = model.W_q(x).reshape(batch, tokens, model.n_heads, model.d_k).transpose(1, 2).contiguous()
    k = model.W_k(x).reshape(batch, tokens, model.n_heads, model.d_k).transpose(1, 2).contiguous()
    v = model.W_v(x).reshape(batch, tokens, model.n_heads, model.d_k).transpose(1, 2).contiguous()
    return q, k, v


def _local_attention(model, inputs, torch):
    x = inputs[0]
    q, k, v = _project_heads(model, x)
    mask = model.create_local_mask(x.shape[1], model.window_size, x.device)[None, None]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (model.d_k ** 0.5)
    expected = torch.matmul(torch.softmax(scores.masked_fill(mask == 0, -1e9), dim=-1), v)
    return {"q": q, "k": k, "v": v, "window_size": model.window_size}, expected


def _longformer_attention(model, inputs, torch):
    x = inputs[0]
    q, k, v = _project_heads(model, x)
    mask = model.create_longformer_mask(
        x.shape[1], model.window_size, model.global_attention_indices, x.device
    )[None, None]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (model.d_k ** 0.5)
    expected = torch.matmul(torch.softmax(scores.masked_fill(mask == 0, -1e9), dim=-1), v)
    return {"q": q, "k": k, "v": v, "window_size": model.window_size}, expected


def _halo_attention(model, inputs, torch):
    x = inputs[0]
    batch, channels, height, width = x.shape
    block, halo, heads = model.block_size, model.halo_size, model.heads
    q_input = x.reshape(batch, channels, height // block, block, width // block, block)
    q_input = q_input.permute(0, 2, 4, 3, 5, 1).reshape(-1, block * block, channels)
    functional = torch.nn.functional
    kv_input = functional.unfold(x, kernel_size=block + 2 * halo, stride=block, padding=halo)
    kv_input = kv_input.reshape(batch, channels, -1, kv_input.shape[-1]).permute(0, 3, 2, 1).reshape(-1, (block + 2 * halo) ** 2, channels)
    q = model.to_q(q_input)
    k, v = model.to_kv(kv_input).chunk(2, dim=-1)
    def split_heads(tensor):
        return tensor.reshape(tensor.shape[0], tensor.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(-1, tensor.shape[1], tensor.shape[-1] // heads).contiguous()
    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    q = (q * model.scale).contiguous()
    valid = torch.ones(1, 1, height, width, device=x.device)
    valid = functional.unfold(valid, kernel_size=block + 2 * halo, stride=block, padding=halo)
    windows = valid.shape[-1]
    mask = valid.unsqueeze(0).expand(batch, -1, -1, -1).permute(0, 3, 1, 2)
    mask = mask.reshape(batch * windows, 1, -1).expand(-1, heads, -1).reshape(batch * windows * heads, 1, -1).bool().contiguous()
    similarity = torch.einsum('b i d, b j d -> b i j', q, k)
    attention = similarity.masked_fill(mask, -torch.finfo(similarity.dtype).max).softmax(dim=-1)
    expected = torch.einsum('b i j, b j d -> b i d', attention, v)
    return {"q": q, "k": k, "v": v, "mask": mask}, expected


def _outlook_attention(model, inputs, torch):
    x = inputs[0]
    batch, height, width, channels = x.shape
    h = (height + model.stride - 1) // model.stride
    w = (width + model.stride - 1) // model.stride
    v = model.v_pj(x).permute(0, 3, 1, 2)
    v = model.unfold(v).reshape(batch, model.num_heads, model.head_dim, model.kernel_size ** 2, h * w)
    v = v.permute(0, 1, 4, 3, 2).contiguous()
    attn = model.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    attn = model.attn(attn).reshape(batch, h * w, model.num_heads, model.kernel_size ** 2, model.kernel_size ** 2)
    attn = (model.scale * attn.permute(0, 2, 1, 3, 4)).softmax(-1).contiguous()
    weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(batch, channels * model.kernel_size ** 2, h * w)
    expected = torch.nn.functional.fold(weighted, (height, width), model.kernel_size, padding=model.padding, stride=model.stride)
    return {"attn": attn, "v": v}, expected


def _polarized_weights(model, x, torch, sequential):
    batch, channels, height, width = x.shape
    channel_wv = model.ch_wv(x).reshape(batch, channels // 2, -1)
    channel_wq = model.softmax_channel(model.ch_wq(x).reshape(batch, -1, 1))
    channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
    channel_weight = model.sigmoid(model.ln(model.ch_wz(channel_wz).reshape(batch, channels, 1).permute(0, 2, 1)))
    channel_weight = channel_weight.permute(0, 2, 1).reshape(batch, channels, 1, 1).contiguous()
    channel_out = channel_weight * x
    spatial_input = channel_out if sequential else x
    spatial_wv = model.sp_wv(spatial_input).reshape(batch, channels // 2, -1)
    spatial_wq = model.agp(model.sp_wq(spatial_input)).permute(0, 2, 3, 1).reshape(batch, 1, channels // 2)
    spatial_weight = model.sigmoid(torch.matmul(model.softmax_spatial(spatial_wq), spatial_wv).reshape(batch, 1, height, width)).contiguous()
    expected = spatial_weight * channel_out if sequential else spatial_weight * x + channel_out
    return {"channel_weight": channel_weight, "spatial_weight": spatial_weight}, expected


def _parallel_polarized(model, inputs, torch):
    values, expected = _polarized_weights(model, inputs[0], torch, False)
    values["x"] = inputs[0]
    return values, expected


def _sequential_polarized(model, inputs, torch):
    values, expected = _polarized_weights(model, inputs[0], torch, True)
    values["x"] = inputs[0]
    return values, expected


def _s2_attention(model, inputs, torch):
    op = model.s2_attention
    x = inputs[0].permute(0, 2, 3, 1)
    channels = x.shape[-1]
    projected = op.mlp1(x)
    # Reproduce the two reference shifts on clones because both update in place.
    x1 = projected[..., :channels].clone()
    x1[:, 1:, :, :channels // 4] = x1[:, :-1, :, :channels // 4]
    x1[:, :-1, :, channels // 4:channels // 2] = x1[:, 1:, :, channels // 4:channels // 2]
    x1[:, :, 1:, channels // 2:channels * 3 // 4] = x1[:, :, :-1, channels // 2:channels * 3 // 4]
    x1[:, :, :-1, channels * 3 // 4:] = x1[:, :, 1:, channels * 3 // 4:]
    x2 = projected[..., channels:2 * channels].clone()
    x2[:, :, 1:, :channels // 4] = x2[:, :, :-1, :channels // 4]
    x2[:, :, :-1, channels // 4:channels // 2] = x2[:, :, 1:, channels // 4:channels // 2]
    x2[:, 1:, :, channels // 2:channels * 3 // 4] = x2[:, :-1, :, channels // 2:channels * 3 // 4]
    x2[:, :-1, :, channels * 3 // 4:] = x2[:, 1:, :, channels * 3 // 4:]
    x3 = projected[..., 2 * channels:].clone()
    x_all = torch.stack([x1, x2, x3], dim=1).contiguous()
    flat = x_all.reshape(x_all.shape[0], 3, -1, channels)
    pooled = flat.sum(1).sum(1)
    split = op.split_attention
    attn = split.softmax(split.mlp2(split.gelu(split.mlp1(pooled))).reshape(x_all.shape[0], 3, channels)).contiguous()
    expected = (attn.unsqueeze(-2) * flat).sum(1).reshape(x_all.shape[0], x_all.shape[2], x_all.shape[3], channels)
    return {"attn": attn, "x_all": x_all}, expected


def _sk_attention(model, inputs, torch):
    conv_outs = [conv(inputs[0]) for conv in model.convs]
    feats = torch.stack(conv_outs, dim=0).contiguous()
    pooled = sum(conv_outs).mean(-1).mean(-1)
    reduced = model.fc(pooled)
    weights = [fc(reduced).reshape(inputs[0].shape[0], inputs[0].shape[1], 1, 1) for fc in model.fcs]
    attn = model.softmax(torch.stack(weights, dim=0)).contiguous()
    return {"attn": attn, "feats": feats}, (attn * feats).sum(0)


def _shuffle_attention(model, inputs, torch):
    x = inputs[0]
    batch, channels, height, width = x.shape
    grouped = x.reshape(batch * model.G, -1, height, width)
    x0, x1 = [part.contiguous() for part in grouped.chunk(2, dim=1)]
    gate_c = model.sigmoid(model.cweight * model.avg_pool(x0) + model.cbias).contiguous()
    s_norm = model.gn(x1).contiguous()
    out = torch.cat([x0 * gate_c, x1 * model.sigmoid(model.sweight * s_norm + model.sbias)], dim=1)
    expected = model.channel_shuffle(out.reshape(batch, -1, height, width), 2)
    return {
        "x0": x0, "x1": x1, "gate_c": gate_c, "s_norm": s_norm,
        "sweight": model.sweight.contiguous(), "sbias": model.sbias.contiguous(),
        "B": batch, "C": channels,
    }, expected


def _fused_mhc(model, inputs, torch):
    return {
        "phi": model.phi_params.contiguous(),
        "bias": model.bias_params.contiguous(),
        "scale": model.rms_norm.scale.contiguous(),
        "alpha_pre": model.alpha_pre.reshape(1).contiguous(),
        "alpha_post": model.alpha_post.reshape(1).contiguous(),
        "alpha_res": model.alpha_res.reshape(1).contiguous(),
        "iters": model.sinkhorn_knopp.iterations,
        "eps_rms": model.rms_norm.eps,
    }, model(inputs[0])


def _mhc_block2d(model, inputs, torch):
    x = inputs[0]
    out = model.bn2(model.conv2(model.relu(model.bn1(model.conv1(x))))).contiguous()
    map_w = model.mapping_projection.weight.contiguous()
    map_bias = (model.mapping_projection.bias + model.mapping_bias).contiguous()
    return {
        "x": x.to(torch.float16).contiguous(),
        "out": out.to(torch.float16).contiguous(),
        "map_w": map_w, "map_bias": map_bias,
        "num_streams": model.num_streams, "sinkhorn_iter": model.sinkhorn_iter,
        "sinkhorn_eps": model.sinkhorn_eps, "sinkhorn_temp": model.sinkhorn_temperature,
        "in_channels": model.in_channels, "out_channels": model.out_channels,
        "height": x.shape[2], "width": x.shape[3],
    }, model(x)


def _mhc_block_bottleneck2d(model, inputs, torch):
    x = inputs[0]
    out = model.relu(model.bn1(model.conv1(x)))
    out = model.relu(model.bn2(model.conv2(out)))
    out = model.bn3(model.conv3(out)).contiguous()
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
    mapping = (model.mapping_projection(pooled) + model.mapping_bias)
    mapping = mapping.reshape(x.shape[0], model.num_streams, model.num_streams).contiguous()
    return {
        "out_bn3": out, "identity": x.contiguous(), "mapping_logits": mapping,
        "sinkhorn_iter": model.sinkhorn_iter, "sinkhorn_eps": model.sinkhorn_eps,
        "sinkhorn_temperature": model.sinkhorn_temperature,
    }, model(x)


def _orthostochastic_project(model, inputs, torch):
    a, b, c = model.coeffs
    return {"steps": model.steps, "eps": model.eps, "a": a, "b": b, "c": c}, model(inputs[0])


ADAPTERS = {
    "AxialAttentionCustom": _axial_attention,
    "BAMCustom": _bam,
    "BlockSparseAttentionCustom": _block_sparse_attention,
    "CBAMBlockCustom": _cbam,
    "GEMMCustom": _gemm,
    "GatedChannelTransformCustom": _gated_channel_transform,
    "PagedAttentionKVCacheCustom": _paged_attention,
    "ParNetAttentionCustom": _parnet,
    "ResidualAttentionCustom": _residual_attention,
    "SEAttentionCustom": _se_attention,
    "SRMCustom": _srm,
    "TripletAttentionCustom": _triplet_attention,
    "CoTAttentionCustom": _cot_attention,
    "CoordAttCustom": _coord_att,
    "CrissCrossAttentionCustom": _criss_cross_attention,
    "DAModuleCustom": _da_module,
    "GCModuleCustom": _gc_module,
    "HaloAttentionCustom": _halo_attention,
    "LocalAttentionCustom": _local_attention,
    "LongformerAttentionCustom": _longformer_attention,
    "OutlookAttentionCustom": _outlook_attention,
    "ParallelPolarizedSelfAttentionCustom": _parallel_polarized,
    "S2AttentionCustom": _s2_attention,
    "SKAttentionCustom": _sk_attention,
    "SequentialPolarizedSelfAttentionCustom": _sequential_polarized,
    "ShuffleAttentionCustom": _shuffle_attention,
    "FusedMhcKernelsCustom": _fused_mhc,
    "MhcBlock2dCustom": _mhc_block2d,
    "MhcBlockBottleneck2dCustom": _mhc_block_bottleneck2d,
    "OrthostochasticProjectCustom": _orthostochastic_project,
}

# These exported operators replace an internal model subgraph.  Passing only
# get_inputs() to them is invalid even when a parameter happens to have a
# similar name.  Keep this list explicit so missing contracts fail before the
# expensive OPP build and are never reported as Kernel numerical failures.
SUBGRAPH_OPERATORS = {
    "AxialAttentionCustom", "BAMCustom", "BlockSparseAttentionCustom",
    "CBAMBlockCustom", "CoTAttentionCustom", "CoordAttCustom",
    "CrissCrossAttentionCustom", "DAModuleCustom", "GCModuleCustom",
    "GEMMCustom", "GatedChannelTransformCustom", "HaloAttentionCustom",
    "LocalAttentionCustom", "LongformerAttentionCustom",
    "OutlookAttentionCustom", "PagedAttentionKVCacheCustom",
    "ParNetAttentionCustom", "ParallelPolarizedSelfAttentionCustom",
    "ResidualAttentionCustom", "S2AttentionCustom", "SEAttentionCustom",
    "SKAttentionCustom", "SRMCustom",
    "SequentialPolarizedSelfAttentionCustom", "ShuffleAttentionCustom",
    "TripletAttentionCustom",
    "MhcBlockBottleneck2dCustom",
}


def adapter_required(name):
    return name in SUBGRAPH_OPERATORS


def adapter_available(name):
    return name in ADAPTERS


def prepare_subgraph(name, model, inputs, torch):
    adapter = ADAPTERS.get(name)
    if adapter is None:
        return None
    with torch.inference_mode():
        values, expected = adapter(model, inputs, torch)
    return values, expected
