rm opinion-q-new-$1-test.log

nohup python test.py\
    --reset_env=True\
    --batch_num=1\
    --train_method='q_new'\
    --evil_nodes_type=$1\
    --tolerance=0.05\
    --save_csv=True\
    --with_noise=True\
    --directed_graph=False\
    > opinion-q-new-$1-test.log &