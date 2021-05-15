import pretty_errors
import torch
from meta_net import meta_dnn
from data_set import message_gen


def maml(args,iter_in_sampled_device,net,current_channel,Noise):
    net.zero_grad()
    para_list_from_net=list(map(lambda p:p[0],zip(net.parameters())))   #怎么还压缩了
    net_meta_intermediate=meta_dnn(if_relu=args.if_relu)
    
    for inner_loop in range(args.num_meta_local_updates):   #这个的默认值为1啊
        if inner_loop==0:   #这句是要执行的
            
            m,label=message_gen(args.bit_num,args.mb_size_meta_train)
            m=m.type(torch.FloatTensor).to(args.device)
            label=label.type(torch.LongTensor).to(args.device)
            #函数就是meta_dnn的激活函数为relu 返回的是一个net类
            out = net_meta_intermediate(m, para_list_from_net, args.if_bias, 
                                        current_channel, args.device, Noise, args.if_RTN)
            
            loss = torch.nn.functional.cross_entropy(out, label)
            


    return iter_in_sampled_device,first_loss_curr,second_loss_curr




def multi_task_learning(args,net,h_list_meta,writer_meta_training,Noise):
    #声明优化器及学习率
    meta_optimiser=torch.optim.Adam(net.parameters(),args.lr_meta_update)   
    h_list_train=h_list_meta[:args.num_channels_meta]

    for epochs in range(1):    #元训练轮数10000次 args.num_epochs_meta_train
        print('外层循环开始')
        first_loss=0
        second_loss=0
        iter_in_sampled_device=0    #平均元设备
        for ind_meta_dev in range(1):   #每次元更新任务数10000次 计算梯度 args.tasks_per_metaupdate
            print('内层循环开始')
            channel_list_total=torch.randperm(len(h_list_train))    #采样
            current_channel_ind=channel_list_total[ind_meta_dev]    #当前信道索引
            current_channel=h_list_train[current_channel_ind]
            print('maml前期完成')   
            if args.if_joint_training:  #联合训练 此句不执行
                iter_in_sampled_device,first_loss_curr,second_loss_curr=joint_training(args,iter_in_sampled_device,
                                                                                    net,current_channel,Noise)
            else:
                iter_in_sampled_device,first_loss_curr,second_loss_curr=maml(args,iter_in_sampled_device,
                                                                                   net,current_channel,Noise)

                print('maml开始')




