class Config(object):
    env = 'defalut'
    checkpoints_path = 'checkpoints'
    save_interval = 10
    train_batch_size = 32
    test_batch_size = 32
    optimizer = 'sgd'
    num_workers = 4
    print_freq = 100
    max_epoch = 30
    lr = 3e-4
    lr_step = 2
    lr_decay = 0.95
    weight_decay = 5e-4
    train_csv = '/opt/ml/input/data/train/train_csv'
    train_img = '/opt/ml/input/data/train/new_img/'
    test_csv = '/opt/ml/input/data/eval/train.csv'
    test_img = '/opt/ml/input/data/eval/new_img/'
