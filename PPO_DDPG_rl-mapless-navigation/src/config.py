dic_agent_conf = {
    "STATE_DIM": 16,
    "ACTOR_LEARNING_RATE": 0.0001,#1e-3,
    "CRITIC_LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "PATIENCE": 10,
    "NUM_LAYERS": 2,
    #"D_DENSE": 32,
    "ACTOR_LOSS": "Clipped",  # or "KL-DIVERGENCE"
    "CLIPPING_LOSS_RATIO": 0.1,
    "ENTROPY_LOSS_RATIO": 0.2,
    "CRITIC_LOSS": "mean_squared_error",
    "OPTIMIZER": "Adam",
    "TARGET_UPDATE_ALPHA": 0.9,
}

dic_env_conf = {
    "ENV_NAME": "LunarLander-v2",
    "GYM_SEED": 1,
    "LIST_STATE_NAME": ["state"],
    "ACTION_RANGE": "-1-1", # or "-1~1"
    "POSITIVE_REWARD": True
}

dic_path ={
    "PPO": "records/PPO/"
}

dic_exp_conf = {
    "TRAIN_ITERATIONS": 100,
    "MAX_EPISODE_LENGTH": 1000,
    "TEST_ITERATIONS": 10
}