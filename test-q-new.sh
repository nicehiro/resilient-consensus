rm opinion-q-new-$1-test.log

nohup python test.py\
    --reset_env=True\
    --batch_num=1000\
    --train_method='q_new'\
    --evil_nodes_type=$1\
    --tolerance=10\
    --save_csv=False\
    --with_noise=False\
    > opinion-q-new-$1-test.log &