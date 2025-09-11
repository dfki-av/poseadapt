_base_ = [
    "./rtmpose-t_8xb32-10e_poseadapt-mode-g_256x192.py",
]


################################### Continual Learning Overrides #####################################
custom_hooks = [
    # CL Strategy: Online Elastic Weight Consolidation (EWC)
    dict(type="EWCPlugin", mode="online", alpha=0.5),
    # Evolution Plugin (required)
    dict(
        type="DefaultEvolutionPlugin",
        mode="last",
        model_cfgs={
            # model cfgs to be updated during evolution
            # in this case, model architecture remains unchanged
        },
    ),
]
