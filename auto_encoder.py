import pretty_errors
import torch.nn as nn

class basic_DNN(nn.Module):
    def __init__(self,M,num_neurons_encoder,n,n_inv_filter,num_neurons_decoder,if_bias,if_relu,if_RTN):
        super(basic_DNN,self).__init__()
        self.enc_fc1=nn.Linear(M,num_neurons_encoder,bias=if_bias)
        self.enc_fc2=nn.Linear(num_neurons_decoder,n,bias=if_bias)

        nun_inv_filter=n_inv_filter*2   #这里就为6
        
        if if_RTN:  #这个为false 不执行 这个if else啥用没有
            self.rtn_1 = nn.Linear(n, n, bias=if_bias)
            self.rtn_2 = nn.Linear(n, n, bias=if_bias)
            self.rtn_3 = nn.Linear(n, num_inv_filter, bias=if_bias)
        else:
            pass

        self.dec_fc1 = nn.Linear(n, num_neurons_decoder, bias=if_bias)
        self.dec_fc2 = nn.Linear(num_neurons_decoder, M, bias=if_bias)

        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()
        
    
    def forward(self,x,h,noise_dist,device,if_RTN):
        x=self.enc_fc1(x)   #这都是线性啊
        x=self.active(x)
        x=self.enc_fc2(x)
        #归一化
        x_norm=torch.norm(x,dim=1)
        x_norm=x_norm.unsqueeze(1)
        x=pow(x.shape[1],0.5)*pow(0.5,0.5)*x/x_norm
        #信道
        x=complex_mul_taps(h,x)
        x=x.to(device)
        #噪声
        n=torch.zeros(x.shape[0],x.shape[1])
        print(x.shape[0])
        for noise_batch_ind in range(x.shape[0]):   #这个range是干嘛的呢
            n[noise_batch_ind]=noise_dist.sample()  #噪声采样
        n=n.type(torch.FloatTensor).to(device)
        x=x+n   #加入噪声

         # RTN
        if if_RTN:
            h_inv = self.rtn_1(x)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_2(h_inv)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_3(h_inv) # no activation for the final rtn (linear activation without weights)
            x = complex_conv_transpose(h_inv, x)
            x = x.to(device)
        else:
            pass

        x = self.dec_fc1(x)
        x = self.activ(x)
        x = self.dec_fc2(x) # softmax taken at loss function
        return x






def dnn(**kwagrs):
    net=basic_DNN(**kwagrs)
    return net













