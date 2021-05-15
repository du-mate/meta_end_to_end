import pretty_errors
import argparse
import torch 
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from auto_encoder import dnn
from data_set import channel_set_gen
import pickle
from meta_train import multi_task_learning


def parse_args():
    parser = argparse.ArgumentParser(description='end_to_end-meta')

    # bit num (k), channel uses (n), tap number (L), number of pilots (P), Eb/N0
    parser.add_argument('--bit_num', type=int, default=4, help='number of bits')
    parser.add_argument('--channel_num', type=int, default=4, help='number of channel uses')
    parser.add_argument('--tap_num', type=int, default=3, help='..')
    parser.add_argument('--mb_size', type=int, default=16, help='minibatch size')
    parser.add_argument('--mb_size_meta_train', type=int, default=16,
                        help='minibatch size during meta-training (this can be useful for decreasing pilots)')
    parser.add_argument('--mb_size_meta_test', type=int, default=16,
                        help='minibatch size for query set (this can be useful for decreasing pilots)')
    parser.add_argument('--Eb_over_N_db', type=float, default=15,
                        help='energy per bit to noise power spectral density ratio')

    # paths
    parser.add_argument('--path_for_common_dir', dest='path_for_common_dir',
                        default='default_folder/default_subfolder/', type=str)
    parser.add_argument('--path_for_meta_training_channels', dest='path_for_meta_training_channels', default=None,
                        type=str)
    parser.add_argument('--path_for_test_channels', dest='path_for_test_channels', default=None, type=str)
    parser.add_argument('--path_for_meta_trained_net', dest='path_for_meta_trained_net', default=None, type=str)

    # neural network architecture (number of neurons for hidden layer)
    parser.add_argument('--num_neurons_encoder', type=int, default=None, help='number of neuron in hidden layer in encoder')
    parser.add_argument('--num_neurons_decoder', type=int, default=None, help='number of neuron in hidden layer in decoder')
    # whether to use bias and relu (if not relu: tanh)
    parser.add_argument('--if_not_bias', dest='if_bias', action='store_false', default=True)
    parser.add_argument('--if_not_relu', dest='if_relu', action='store_false', default=True)
    # RTN
    parser.add_argument('--if_RTN', dest='if_RTN', action='store_true', default=False)
    # in case of running on gpu, index for cuda device
    parser.add_argument('--cuda_ind', type=int, default=0, help='index for cuda device')

    # experiment details (hyperparameters, number of data for calculating performance and for meta-training
    parser.add_argument('--lr_testtraining', type=float, default=0.001, help='lr for adaptation to new channel')
    parser.add_argument('--lr_meta_update', type=float, default=0.01, help='lr during meta-training: outer loop (update initialization) lr')
    parser.add_argument('--lr_meta_inner', type=float, default=0.1, help='lr during meta-training: inner loop (local adaptation) lr')
    parser.add_argument('--test_size', type=int, default=1000000, help='number of messages to calculate BLER for test (new channel)')
    parser.add_argument('--num_channels_meta', type=int, default=100, help='number of meta-training channels (K)')
    parser.add_argument('--num_channels_test', type=int, default=20, help='number of new channels for test (to get average over BLER)')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=20, help='number of meta-training channels considered in one meta-update')
    parser.add_argument('--num_meta_local_updates', type=int, default=1, help='number of local adaptation in meta-training')
    parser.add_argument('--num_epochs_meta_train', type=int, default=10000,
                        help='number epochs for meta-training')

    # if run for joint training, if false: meta-learning
    parser.add_argument('--if_joint_training', dest='if_joint_training', action='store_true', default=False) # else: meta-learning for multi-task learning
    # whether to use Adam optimizer to adapt to a new channel
    parser.add_argument('--if_not_test_training_adam', dest='if_test_training_adam', action='store_false',
                        default=True)
    # if run on toy example (Fig. 2 and 3)
    parser.add_argument('--if_toy', dest='if_toy', action='store_true',
                        default=False)
    # to run on a more realistic example (Fig. 4)
    parser.add_argument('--if_RBF', dest='if_RBF', action='store_true',
                        default=False)
    parser.add_argument('--test_per_adapt_fixed_Eb_over_N_value', type=int, default=15,
                        help='Eb/N0 in db for test')
    # desinged for maml: sgd during args.num_meta_local_updates with args.lr_meta_inner and then follow Adam optimizer with args.lr_testtraining
    parser.add_argument('--if_adam_after_sgd', dest='if_adam_after_sgd', action='store_true',
                        default=False)

    args = parser.parse_args()

    args.device = torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")
    if args.num_neurons_encoder == None: # unless specified, set number of hidden neurons to be same as the number of possible messages
        args.num_neurons_encoder = pow(2,args.bit_num)
    if args.num_neurons_decoder == None:
        args.num_neurons_decoder = pow(2, args.bit_num)

    if args.if_test_training_adam == False:
        args.if_adam_after_sgd = False

    if args.if_toy == True:
        print('running for toy scenario')
        args.bit_num = 2
        args.channel_num = 1
        args.tap_num = 1
        args.mb_size = 4
        args.mb_size_meta_train = 4
        args.mb_size_meta_test = 4
        args.num_channels_meta = 20
        args.num_neurons_encoder = 4
        args.num_neurons_decoder = 4
    elif args.if_RBF == True:
        print('running for a more realistic scenario')
        args.bit_num = 4
        args.channel_num = 4
        args.tap_num = 3
        args.mb_size = 16
        args.mb_size_meta_train = 16
        args.mb_size_meta_test = 16
        args.num_channels_meta = 100
        args.num_neurons_encoder = 16
        args.num_neurons_decoder = 16
    else:
        print('running on custom environment')
    # print('Running on device: {}'.format(args.device))
    return args


if __name__=='__main__':
    args=parse_args()
    # print('called with args:')
    # print(args)

    curr_time=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") #这个时间为后面保存数据
    common_dir='./'+args.path_for_common_dir+curr_time+'/'


    PATH_before_adapt=common_dir+'saved_model/'+'before_adapt/'+'init_net'
    #路径
    PATH_meta_intermediate=common_dir+'saved_model/'+'during_meta_training/'+'epochs/'

    #文件夹 训练时记得取消注释
    os.makedirs(common_dir+'saved_model/'+'before_adapt/')
    # os.makedirs(common_dir + 'saved_model/' + 'after_adapt/')
    # os.makedirs(PATH_meta_intermediate)
    os.makedirs(common_dir + 'meta_training_channels/')
    # os.makedirs(common_dir + 'test_channels/')
    # os.makedirs(common_dir + 'test_result/')

    dir_meta_training=common_dir+'TB/'+'meta_training'
    writer_meta_training=SummaryWriter(dir_meta_training)   #写入文件夹
    dir_during_adapt=common_dir+'TB/'+'during_adapt/'

    test_Eb_over_N_range=[args.test_per_adapt_fixed_Eb_over_N_value]    #列表就一个值15
    test_adapt_range=[0,1,2,5,10,100,200,1000,10000]

    if len(test_Eb_over_N_range)>1: #这条不执行
        assert len(test_adapt_range)==1
    if len(test_adapt_range)>1: #此条语句执行
        assert len(test_Eb_over_N_range)==1


    test_result_all_PATH=common_dir+'test_result/'+'test_result.mat'
    save_test_result_dict={}

    actual_channel_num=args.channel_num*2   #channel_num=4 

    #bit_num=4 num_neurons_encoder无默认值 tap_num=3 if_bias默认为true if_relu默认为true if_RTN为false
    net=dnn(M=pow(2,args.bit_num),num_neurons_encoder=args.num_neurons_encoder,n=actual_channel_num,
            n_inv_filter=args.tap_num,num_neurons_decoder=args.num_neurons_decoder,if_bias=args.if_bias,
            if_relu=args.if_relu,if_RTN=args.if_RTN)

    net_for_testtraining = dnn(M=pow(2, args.bit_num), num_neurons_encoder=args.num_neurons_encoder, 
                                n=actual_channel_num, n_inv_filter = args.tap_num,num_neurons_decoder=args.num_neurons_decoder,
                                if_bias=args.if_bias, if_relu=args.if_relu, if_RTN=args.if_RTN)
   
    Eb_over_N=pow(10,(args.Eb_over_N_db/10))    #Eb_over_N_db=15
    R=args.bit_num/args.channel_num #bit_num=4 channel_num=4
    noise_var=1/(2*R*Eb_over_N) #噪声方差
    Noise=torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(actual_channel_num),noise_var*torch.eye(actual_channel_num))

    if args.path_for_meta_training_channels is None:    #这句是要执行的
        print('generate meta-training channels')
        #num_channels_meta=100 tap_num=3 if_toy默认false
        h_list_meta=channel_set_gen(args.num_channels_meta,args.tap_num,args.if_toy)
        #这个路径干嘛哦
        h_list_meta_path=common_dir+'meta_training_channels/'+'training_channels.pckl'
        f_meta_channels=open(h_list_meta_path,'wb')
        torch.save(h_list_meta,f_meta_channels)    #前一个列表 后一个路径 相当于是保存
        f_meta_channels.close() #关闭文件夹
    else:
        print('load previous generated channels')
        h_list_meta_path=args.path_for_meta_training_channels+'/'+'training_channels.pckl'
        f_meta_channels=open(h_list_meta_path,'rb')
        h_list_meta=pickle.load(f_meta_channels)
        f_meta_channels.close()

    if args.path_for_meta_trained_net is None:  #这句要执行的
        if args.if_joint_training:  #默认false 此句不执行
            print('start joint training')
        else:
            print('start meta training')    #这一步是执行了的
        multi_task_learning(args,net,h_list_meta,writer_meta_training,Noise)
        # torch.save(net.state_dict(),PATH_before_adapt)


