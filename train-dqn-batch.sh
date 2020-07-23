rm -r logs/
rm opinion.log
nohup python test.py --episodes='10000000'\
               --epochs=1000\
               --restore=False\
               --need_exploit=True\
               --batch_size=64\
               --memory_size='1000000'\
               --train=True\
               --lr=0.00001\
               --hidden_size=256\
               --hidden_layer=4\
               --log=True\
               --train_method='dqn_train'\
    > opinion.log &
# nohup python test.py > opinion.log &