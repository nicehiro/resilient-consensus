rm rc-q-consensus-dynamic-$1-test.log

nohup python test.py\
    --reset_env=True\
    --batch_num=100\
    --train_method='dynamic_q_consensus'\
    --evil_nodes_type=$1\
    --tolerance=0.05\
    --save_csv=False\
    --with_noise=True\
    --directed_graph=True\
    > rc-q-consensus-dynamic-$1-test.log &