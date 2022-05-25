# Generate full trajectory
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/gen_full_traj.py --exp_name loop_2layer_clip_adam_gradeps0.2_lr1e-2 --n_hidden_layer 2 

# (Oracle?) Linear combination of sigma_1, simga_2, sigma_1**2, sigma_2**2, sigma1 * sigma2, (sigma1 * sigma2)**2
    # SGD
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_basis_sgd_lr3 --use_loop --optimizer SGD --Psi_lr 3

    # Adam
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_basis_adam_lr1 --use_loop --optimizer Adam --Psi_lr 1 --psi_model_input_type basis
    

# Compare (0 hidden layer)

    # 1. Hao's gradient norm trusting strategy
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_noclip_sgd_gradeps0.2_lr3 --use_loop --optimizer SGD --Psi_lr 3 --grad_eps 2e-1

    # 2. Grad clipping
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_clip_sgd_lr3 --use_loop --optimizer SGD --Psi_lr 3 --clip_grad 1e-1

    # 3. Both 1 and 2
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_clip_sgd_gradeps0.2_lr3 --use_loop --optimizer SGD --Psi_lr 3 --grad_eps 2e-1 --clip_grad 1e-1

    # 4. None
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_noclip_sgd_lr3 --use_loop --optimizer SGD --Psi_lr 3 --n_hidden_layer 0

    # 5. Use first 50 frames (compute loss at frame 10, 20, 30, 40, 50), but evaluate at frame 100
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_0layer_noclip_sgd_sup5eval10_lr10 --use_loop --optimizer SGD --Psi_lr 10 --supervise_clip_len 50


# Compare (2 hidden layer)

    # 1. Hao's gradient norm trusting strategy
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_2layer_noclip_sgd_gradeps0.2_lr3 --use_loop --optimizer SGD --Psi_lr 3 --grad_eps 2e-1

    # 2. Grad clipping
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_2layer_clip_sgd_lr1 --use_loop --optimizer SGD --Psi_lr 1 --clip_grad 1e-1 --n_hidden_layer 2

    # 3. Both 1 and 2
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_2layer_clip_sgd_gradeps0.2_lr1 --use_loop --optimizer SGD --Psi_lr 1 --grad_eps 2e-1 --clip_grad 1e-1 --n_hidden_layer 2

    # 4. None
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_2layer_noclip_sgd_lr1 --use_loop --optimizer SGD --Psi_lr 1 --n_hidden_layer 2

    # 6. Both 1 and 2, Adam
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 DISPLAY=:20 python learnable/learn_Psi/scripts/one_clip_train.py --exp_name loop_2layer_clip_adam_gradeps0.2_lr1e-2 --use_loop --optimizer Adam --Psi_lr 1e-2 --grad_eps 2e-1 --clip_grad 1e-1 --n_hidden_layer 2