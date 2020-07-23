rm -r logs/
rm opinion-dqn-single-step.log
nohup python test.py --episodes='2000'\
                     --epochs=2\
                     --restore=False\
                     --need_exploit=False\
                     --batch_size=2\
                     --memory_size='2'\
                     --train=True\
                     --lr=0.0001\
                     --hidden_size=256\
                     --hidden_layer=4\
                     --log=True\
                     --train_method='dqn_train'\
                     --reset_env=False\
                     --batch_num=10000\
    > opinion-dqn-single-step.log &