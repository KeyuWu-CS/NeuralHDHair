import torch
def pixel_norm(x, epsilon=1e-8):
        return x * torch.rsqrt(torch.sum(x**2, dim=1,keepdim=True) + epsilon)  # rsqrt = 1./sqrt



def instance_norm_video(x, label,epsilon=1e-8):
    size=x.size(-1)
    x=x.view(-1,1,size,size)
    label=label.view(-1,1,size,size)
    assert len(x.size()) == 4 # NCHW

    hair=x*label
    mean = torch.sum(hair, dim=[0,2, 3], keepdim=True) / torch.sum(label, dim=[0, 2, 3], keepdim=True)
    square=torch.where(label==1,(hair-mean)**2,torch.zeros_like(hair))
    var=torch.sum(square,dim=[0,2,3],keepdim=True)/torch.sum(label,dim=[0,2,3],keepdim=True)
    # out=torch.where(label==1,(x-mean)*torch.rsqrt(var+epsilon),torch.zeros_like(x))
    out=(x-mean)*torch.rsqrt(var+epsilon)
    if torch.sum(torch.isnan(out)) > 0:
        print('NAN_1')
    if torch.sum(torch.isinf(out)) > 0:
        print('INF_1')
    return out


