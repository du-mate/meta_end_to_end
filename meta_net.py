import pretty_errors
import torch.nn as nn
from torch.nn import functional as F
import torch
from funcs import complex_mul_taps


class meta_Net_DNN(nn.Module):
    def __init__(self,if_relu): #仅从其它网络获得参数
        super(meta_Net_DNN,self).__init__()
        if if_relu: #这个是要执行的
            self.activ=nn.ReLU()
        else:
            self.activ=nn.Tanh()
        self.tanh=nn.Tanh()

    def forward(self,x,var,if_bias,h,device,noise_dist,if_RTN): #这个函数是必调用的 定义的是网络结构
        idx_init=0  #idx初始为0
        if if_bias: #执行这个
            gap=2
        else:
            gap=1

        idx=idx_init    #这里出现idx
        while idx<len(var): #len(var)=8
            if idx>idx_init:    #开始无激活
                if idx==gap*2+idx_init:  #后面开始编码
                    pass
                else:
                    x=self.activ(x) #激活函数
                
            if idx==idx_init:
                if if_bias: #执行这句
                    w1,b1=var[idx],var[idx+1]   #权重和偏置
                    x=F.linear(x,w1,b1)
                    idx+=2
                else:
                    w1=var[idx] 
                    x=F.linear(x,w1)
                    idx+=1

            elif idx==gap*1+idx_init:
                if if_bias:
                    w2, b2 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w2, b2)
                    idx += 2
                else:
                    w2 = var[idx]  # weight and bias
                    x = F.linear(x, w2)
                    idx += 1

            elif idx==gap*2+idx_init:   #现在需要归一化并通过信道
                x_norm=torch.norm(x,dim=1)
                x_norm=x_norm.unsqueeze(1)
                x=pow(x.shape[1],0.5)*pow(0.5,0.5)*x/x_norm
                x=complex_mul_taps(h,x) #这一块也是个函数 那就是这个函数没有写好
                x=x.to(device)

                #噪声
                n=torch.zeros(x.shape[0],x.shape[1])
                for noise_batch_ind in range(x.shape[0]):   #这个range又是干嘛的呢
                    n[noise_batch_ind]=noise_dist.sample()
                n=n.type(torch.FloatTensor).to(device)
                x=x+n

                if if_RTN:  #默认false 此句不执行
                    if if_bias:
                        w_rtn_1, b_rtn_1 = var[idx], var[idx+1]
                        h_inv = F.linear(x, w_rtn_1, b_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2, b_rtn_2 = var[idx+2], var[idx + 3]
                        h_inv = F.linear(h_inv, w_rtn_2, b_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3, b_rtn_3 = var[idx + 4], var[idx + 5]
                        h_inv = F.linear(h_inv, w_rtn_3, b_rtn_3)
                        rtn_gap = 6
                    else:
                        w_rtn_1 = var[idx]
                        h_inv = F.linear(x, w_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2 = var[idx+1]
                        h_inv = F.linear(h_inv, w_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3 = var[idx+2]
                        h_inv = F.linear(h_inv, w_rtn_3)
                        rtn_gap = 3
                    x = complex_conv_transpose(h_inv, x)
                    x = x.to(device)
                else:   #此句执行
                    rtn_gap = 0
                ############## from now, demodulator
                if if_bias: #此句执行
                    w3, b3 = var[idx+ rtn_gap], var[idx + rtn_gap + 1]  # weight and bias
                    x = F.linear(x, w3, b3)
                    idx += (2 + rtn_gap)
                else:
                    w3 = var[idx + rtn_gap]  # weight
                    x = F.linear(x, w3)
                    idx += (1 + rtn_gap)

            elif idx==gap*3+rtn_gap+idx_init:
                if if_bias:
                    w4, b4 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w4, b4)
                    idx += 2
                else:
                    w4 = var[idx]  # weight
                    x = F.linear(x, w4)
                    idx += 1
        return x




def meta_dnn(**kwagrs):
    net=meta_Net_DNN(**kwagrs)
    return net