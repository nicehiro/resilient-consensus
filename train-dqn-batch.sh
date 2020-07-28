# rm -r logs/
# rm opinion.log
nohup python test.py --episodes='10000000'\
               --epochs=1000\
               --restore=False\
               --need_exploit=True\
               --batch_size=64\
               --memory_size='1000000'\
               --train=True\
               --lr=0.001\
               --hidden_size=512\
               --hidden_layer=4\
               --log=True\
               --log_path='replay-2r1c'\
               --reset_env=True\
               --batch_num=1\
               --train_method='dqn_train'\
               --save=True\
               --evil_nodes_type='2r1c'\
    > opinion-2r1c.log &
# nohup python test.py > opinion.log &