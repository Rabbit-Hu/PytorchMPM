# CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_noclip_sgd_gradeps_lr10 --use_loop --optimizer SGD --Psi_lr 10 --grad_eps 1e-1

# CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_noclip_adam_gradeps --use_loop --optimizer Adam --Psi_lr 3e-1 --grad_eps 1e-1


# Compare
    # 1. Hao's gradient norm trusting strategy
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_noclip_sgd_gradeps0.2_lr10 --use_loop --optimizer SGD --Psi_lr 10 --grad_eps 2e-1

    # 2. Grad clipping
    CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_clip_sgd_lr10 --use_loop --optimizer SGD --Psi_lr 10 --clip_grad 1e-1

    # 3. Both 1 and 2
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_clip_sgd_gradeps0.2_lr10 --use_loop --optimizer SGD --Psi_lr 10 --grad_eps 2e-1 --clip_grad 1e-1

    # 4. None
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_noclip_sgd_lr10 --use_loop --optimizer SGD --Psi_lr 10