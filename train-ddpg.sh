rm -rf ddpg-$1-logs/
rm opinion-ddpg-$1.log

nohup python test.py\
    --episodes='3000'\
    --epochs=100\
    --restore=False\
    --memory_size='10000'\
    --batch_size=64\
    --actor_lr=0.00001\
    --critic_lr=0.0001\
    --hidden_size=512\
    --hidden_layer=3\
    --log_path=ddpg-$1-logs\
    --reset_env=True\
    --batch_num=1\
    --train_method='ddpg_train'\
    --train=True\
    --save=True\
    --evil_nodes_type=$1\
    --tolerance=0.01\
    --polyak=0.99\
    > opinion-ddpg-$1.log &