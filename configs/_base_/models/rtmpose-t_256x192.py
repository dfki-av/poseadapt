codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

backbone = dict(
    _scope_="mmdet",
    type="CSPNeXt",
    arch="P5",
    expand_ratio=0.5,
    deepen_factor=0.167,
    widen_factor=0.375,
    out_indices=(4,),
    channel_attention=True,
    norm_cfg=dict(type="SyncBN"),
    act_cfg=dict(type="SiLU"),
)

head = dict(
    type="RTMCCHead",
    in_channels=384,
    input_size=codec["input_size"],
    in_featuremap_size=tuple([s // 32 for s in codec["input_size"]]),
    simcc_split_ratio=codec["simcc_split_ratio"],
    final_layer_kernel_size=7,
    gau_cfg=dict(
        hidden_dims=256,
        s=128,
        expansion_factor=2,
        dropout_rate=0.0,
        drop_path=0.0,
        act_fn="SiLU",
        use_rel_bias=False,
        pos_enc=False,
    ),
    loss=dict(
        type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True
    ),
    decoder=codec,
)
