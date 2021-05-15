import pretty_errors
import torch


def complex_mul(h,x):   #h在batch固定 x在batch不同的
    if len(h.shape)==1: #h在所有信息中都一样 
        y=torch.zeros(x.shape[0],2,dtype=torch.float)
        y[:, 0] = x[:, 0] * h[0] - x[:, 1] * h[1]
        y[:, 1] = x[:, 0] * h[1] + x[:, 1] * h[0]
    elif len(h.shape) == 2:
        # h_estimated is not averaged
        assert x.shape[0] == h.shape[0]
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[:, 0] - x[:, 1] * h[:, 1]
        y[:, 1] = x[:, 0] * h[:, 1] + x[:, 1] * h[:, 0]
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    return y





def complex_mul_taps(h,x_tensor):   #h 信道 x_tensor 输入
    if len(h.shape)==1:
        L=h.shape[0]//2 #信道向量//2代表taps
    elif len(h.shape)==2:
        L=h.shape[1]//2 
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    y=torch.zeros(x_tensor.shape[0],x_tensor.shape[1],dtype=torch.float)
    assert x_tensor.shape[1]%2==0  #整除？
    for ind_channel_use in range(x_tensor.shape[1]//2): #这里有个循环
        for ind_conv in range(min(L,ind_channel_use+1)):    #这里有个循环
            if len(h.shape)==1:
                y[:,(ind_channel_use)*2:(ind_channel_use+1)*2]+=complex_mul(h[2*ind_conv:2*(ind_conv+1)],x_tensor[:,(ind_channel_use-ind_conv)*2:(ind_channel_use-ind_conv+1)*2])
            else:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(
                    h[:, 2 * ind_conv:2 * (ind_conv + 1)],
                    x_tensor[:, (ind_channel_use - ind_conv) * 2:(ind_channel_use - ind_conv + 1) * 2])
    return y







