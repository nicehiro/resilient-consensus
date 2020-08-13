rm opinion-maddpg-$1-test.log

nohup python test.py\
    --episodes='1'\
    --epochs=1000\
    --restore=False\
    --memory_size='1000000'\
    --batch_size=1024\
    --actor_lr=0.0001\
    --critic_lr=0.001\
    --hidden_size=512\
    --hidden_layer=3\
    --log_path='maddpg-logs'\
    --reset_env=True\
    --batch_num=1000\
    --train_method='maddpg_train'\
    --train=False\
    --save=True\
    --evil_nodes_type=$1\
    --tolerance=0.05\
    --save_csv=True\
    --with_noise=False\
    > opinion-maddpg-$1-test.log &