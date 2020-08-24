rm -rf qnew-vs-maddpg-logs/
rm opinion-qnew-vs-maddpg.log

nohup python test.py\
    --episodes='300'\
    --epochs=100\
    --restore=False\
    --memory_size='1000000'\
    --batch_size=1024\
    --actor_lr=0.0001\
    --critic_lr=0.001\
    --hidden_size=512\
    --hidden_layer=3\
    --log_path=qnew-vs-maddpg-logs\
    --reset_env=True\
    --batch_num=1\
    --train_method='rival_qnew'\
    --train=True\
    --save=True\
    --evil_nodes_type='maddpg'\
    --polyak=0.99\
    --tolerance=0.05\
    > opinion-qnew-vs-maddpg.log &