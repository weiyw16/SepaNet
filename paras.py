# inverse gradient for interpretating the PS wave separation
#              Author: Yanwen Wei
#              Email: wei_yanwen@163.com
#              Date: 2021-12-28
#


import argparse

class parameters():
    def get_parser(self, debug=0):
        parser = argparse.ArgumentParser(description='PyTorch vsp decomposition Training')
        parser.add_argument('--data_path', metavar='DIR', help='path to dataset')
        parser.add_argument('--test_model_id', default=10, type=int, metavar='N', help='test model id')
        parser.add_argument('--train_model_id', default=10, type=int, metavar='N', help='train model id')
        parser.add_argument('--out_name', metavar='DIR', help='the given name of saved model')
        parser.add_argument('--epoches', default=100, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--epoch_start', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
        parser.add_argument('--batch_size', default=2, type=int, metavar='N', help='batch_size')
        parser.add_argument('--data_start', default=0, type=int, metavar='N', help='start index of dataset')
        parser.add_argument('--max_data', default=100, type=int, metavar='N', help='maximum of data')
        parser.add_argument('--sample_interval', default=10, type=int,metavar='N', help='save model')
        parser.add_argument('--lr', default=0.0002, type=float, metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--beta1', default=0.5, type=float, metavar='Beta', help='beta1')
        parser.add_argument('--lambda_L1', default=100, type=float, metavar='Lambda', help='Lambda')
        parser.add_argument('--lambda_A', default=10, type=float, metavar='Lambda', help='Lambda')
        parser.add_argument('--lambda_B', default=10, type=float, metavar='Lambda', help='Lambda')
        parser.add_argument('--lambda_idt', default=0.5, type=float, metavar='Lambda', help='Lambda')

        parser.add_argument('--nt', default=2001, type=int, metavar='N', help='nt')
        parser.add_argument('--nr', default=467, type=int, metavar='N', help='nr')
        parser.add_argument('--ns', default=151, type=int, metavar='N', help='ns')
        parser.add_argument('--nph', default=3, type=int, metavar='N',help='nph')
        parser.add_argument('--nop', default=1, type=int, metavar='N',help='nop')

        parser.add_argument('--num_workers', default=-1, type=int, help='number of nodes for distributed training')
        parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
        parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
        parser.add_argument('--dist_url', default='tcp://127.0.0.0:12306', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

        parser.add_argument('--num_gpu', default=-1, type=int, help='number of nodes for distributed training')
        parser.add_argument('--gpuid', default=None, nargs='+', type=int, help='GPU id to use.')

        parser.add_argument('--if_continue', action='store_true', help='continue training')
        parser.add_argument('--continue_net_path_G', metavar='DIR', help='path to model')
#         parser.add_argument('--continue_model_prefix_G', type=str, help='the prefix name of the load in model')
        parser.add_argument('--continue_net_path_D', metavar='DIR', help='path to model')
#         parser.add_argument('--continue_model_prefix_D', type=str, help='the prefix name of the load in model')
        parser.add_argument('--data_norm_type', default=0, type=int, help='nomaliza type')
        parser.add_argument('--jsample', default=1, type=int, help='read in interval of datasets')

        ## validation
        parser.add_argument('--data_path_valid', metavar='DIR', help='path to dataset')
        parser.add_argument('--data_start_valid', default=0, type=int,metavar='N', help='start index of dataset')
        parser.add_argument('--max_data_valid', default=100, type=int,metavar='N', help='maximum of data')
        parser.add_argument('--validate_interval', default=10, type=int, metavar='N', help='save model')
        parser.add_argument('--jsample_valid', default=1, type=int, help='read in interval of datasets')
        parser.add_argument('--do_validate', action='store_true', help='validate')

        # inverse
        parser.add_argument('--losscom', default='p', type=str, help='which component to be chosen as loss')


        ## add save dir
        parser.add_argument('--out_predix', default='model_shot', type=str, help='save dir prefix')
        parser.add_argument('--reverse', default='False', type=str, help='whether running the reverse part')

        # weights
        parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')

        parser.add_argument('--save_testout', default=0, type=int, metavar='N', help='if save testout')

        if debug:
            return parser.parse_args([])
        if not debug:
            return parser.parse_args()