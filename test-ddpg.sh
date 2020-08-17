rm opinion-ddpg-$1-test.log

nohup python test.py\
    --episodes='1'\
    --epochs=1000\
    --memory_size='10000'\
    --batch_size=64\
    --actor_lr=0.00001\
    --critic_lr=0.001\
    --hidden_size=256\
    --hidden_layer=3\
    --log_path=ddpg-$1-logs\
    --reset_env=True\
    --batch_num=1\
    --train_method='ddpg_train'\
    --train=False\
    --save=False\
    --evil_nodes_type=$1\
    --tolerance=0.05\
    --polyak=0.95\
    --save_csv=True\
    --with_noise=False\
    > opinion-ddpg-$1-test.log  &