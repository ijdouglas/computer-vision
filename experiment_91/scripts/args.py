import argparse

def setup_args():
    parser = argparse.ArgumentParser(description='Visual Features in Word Learning')
    parser.add_argument('--train_set', default=None)
    parser.add_argument('--test_set', default=None)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--reg', type=float, default=10e-3, metavar='R',
                        help='weight decay')
    ### > start IJD ADDED:
    parser.add_argument('--num_classes', type=int, default=2,
                        help='the number of classes in the response variable')
    parser.add_argument('--label_key', type=str, default=None, 
                        help='The name of the dependent variable/image classes/labels')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Diretory where to save the cross-validation results for all subjects')
    parser.add_argument('--normalization', default='mnist', type=str,
                        help = '`none` = no normalization. `mnist` (default) will subtract (and divide by) the mean (and std) of the MNIST. `imagenet` will do the same for the mean and std of the imagenet dataset')
    ### < end IJD ADDED
    parser.add_argument('--target_number', type=int, default=None, metavar='T',
                        help='bags have a positive labels if they contain at least one 9')
    # parser.add_argument('--num_train', type=int, default=1000, metavar='NTrain',
    #                     help='number of bags in training set')
    # parser.add_argument('--num_test', type=int, default=100, metavar='NTest',
    #                     help='number of bags in test set')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', type=str, default='vgg16', help='set backbone architecture')
    parser.add_argument('--chkpt_dir', default=None, type=str)
    parser.add_argument("--config", default=None)
    parser.add_argument('--gpu','-g', type=int, default=-2,help = "gpu id. -1 means cpu. -2 means use CUDA_VISIBLE_DEVICES one")
    parser.add_argument('--loss_balanced', default=False)
    parser.add_argument('--batch', default=128)
    parser.add_argument('--freeze_backbone', default=False)
    parser.add_argument('--test_at_end', action='store_true', default=False)
    parser.add_argument('--hidden_size', default=1024, type=int)

    return parser.parse_args()

def load_config(args, cfg):
    args.train_set = cfg['train_set']
    args.test_set = cfg['test_set']
    args.root_dir = cfg['root_dir']
    args.epochs = cfg['epochs']
    args.model = cfg['model']

    return args

