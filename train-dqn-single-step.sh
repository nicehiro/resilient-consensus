# rm -r logs/
# rm opinion-dqn-single-step.log
nohup python test.py --episodes='1000'\
                     --epochs=2\
                     --restore=False\
                     --need_exploit=False\
                     --batch_size=2\
                     --memory_size='2'\
                     --train=True\
                     --lr=0.0001\
                     --hidden_size=512\
                     --hidden_layer=4\
                     --log=True\
                     --log_path='single-step-3c'\
                     --train_method='dqn_train'\
                     --reset_env=False\
                     --batch_num=1000\
                     --save=False\
                     --evil_nodes_type='3c'\
    > opinion-dqn-single-step-3c.log &