import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torchvision.models as models
from torch.autograd import Variable
import torch.distributions as td

import numpy as np

from torchvision.ops import roi_pool,roi_align

device='cuda'
base_num=64

class ConvBlock(nn.Module):
    """ConvBlock for UNet"""
    def __init__(self,input_channels,output_channels,max_pool,return_single=False,dropout_rate=0):
        super(ConvBlock,self).__init__()
        self.max_pool=max_pool
        self.dropout_rate=dropout_rate
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.BatchNorm2d(output_channels,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.BatchNorm2d(output_channels,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.ReLU())
        self.return_single=return_single
        if max_pool:
            self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        if self.dropout_rate==0:
            x=self.conv(x)
        else:
            x=F.dropout2d(self.conv[0](x),self.dropout_rate,training=True)
            x=self.conv[1](x)
            x=self.conv[2](x)
            F.dropout2d(self.conv[3](x),self.dropout_rate,training=True)
            x=self.conv[4](x)
            x=self.conv[5](x)
        b=x
        if self.max_pool:
            x=self.pool(x)
        if self.return_single:
            return x
        else:
            return x,b

class DeconvBlock(nn.Module):
    """DeconvBlock for UNet"""
    def __init__(self,input_channels,output_channels,intermediate_channels=-1,dropout_rate=0,use_b=True):
        super(DeconvBlock,self).__init__()
        self.use_b=use_b
        input_channels=int(input_channels)
        output_channels=int(output_channels)
        if intermediate_channels<0:
            intermediate_channels=output_channels*2
        else:
            intermediate_channels=input_channels
        self.upconv=[]
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=intermediate_channels//2,kernel_size=3,stride=1,padding=1))
        if use_b:
            self.conv=ConvBlock(intermediate_channels,output_channels,False,dropout_rate)
        else:
            self.conv=ConvBlock(intermediate_channels//2,output_channels,False,dropout_rate)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        x=self.upconv(x)
        if self.use_b:
            x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x

class ProbDeconvBlock(nn.Module):
    """DeconvBlock for UNet"""
    def __init__(self,input_channels,prob_channels,output_channels):
        super(ProbDeconvBlock,self).__init__()
        input_channels=int(input_channels)
        output_channels=int(output_channels)
        self.prob_channels=prob_channels
        self.upconv=[]
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv=ConvBlock(2*output_channels+prob_channels,output_channels,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b,p):
        x=self.upconv(x)
        if self.prob_channels!=0:
            x=torch.cat((x,b,p),dim=1)
        else:
            x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x



class Encoder(nn.Module):
    """Encoder for both UNet and UNet transformer"""
    def __init__(self,input_channels,num_layers,base_num,dropout_rate=0):
        super(Encoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True,dropout_rate=dropout_rate))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1),dropout_rate=dropout_rate))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b

class AEEncoder(nn.Module):
    def __init__(self,input_channels,num_layers,base_num,dropout_rate=0):
        super(AEEncoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True,dropout_rate=dropout_rate))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1),dropout_rate=dropout_rate))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        for i in range(self.num_layers):
            x,_=self.conv[i](x)
        return x

class AEDecoder(nn.Module):
    def __init__(self,num_layers,base_num,dropout_rate=0):
        super(AEDecoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1)),dropout_rate,use_b=False))
        self.conv=nn.Sequential(*self.conv)
    def forward(self,x):
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                x=self.conv[i](x,None)
        return x

class Decoder(nn.Module):
    """Decoder for UNet"""
    def __init__(self,num_classes,num_layers,base_num,include_last=True,dropout_rate=0):
        super(Decoder,self).__init__()
        self.include_last=include_last
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1)),dropout_rate))
        if include_last:
            self.conv.append(nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=1,stride=1,padding=0))
        self.conv=nn.Sequential(*self.conv)
    def forward(self,x,b):
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                x=self.conv[i](x,b[i])
            else:
                if self.include_last:
                    x=self.conv[i](x)
        return x


class ReshapedDistribution(td.Distribution):
    def __init__(self, base_distribution, new_event_shape):
        #super().__init__(batch_shape=base_distribution.batch_shape, event_shape=new_event_shape)
        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape

    @property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return self.base_distribution.arg_constraints()

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def mean_not_reshaped(self):
        return self.base_distribution.mean

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self,sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(sample_shape+self.new_shape)

    def log_prob(self,value):
        return self.base_distribution.log_prob(value.view(self.batch_shape+(-1,)))

    def entropy(self):
        return self.base_distribution.entropy()

    @property
    def cov_factor(self):
        return self.base_distribution._unbroadcasted_cov_factor

    @property
    def cov_diag(self):
        return self.base_distribution._unbroadcasted_cov_diag

    @property
    def capacitance_tril(self):
        try:
            return self.base_distribution._capacitance_tril
        except:
            return torch.zeros((1,1,1,1))


class UNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,mode='softmax',base_num=64,rank=10,epsilon=0.00001,diagonal=False,dropout_rate=0):
        super(UNet,self).__init__()
        self.dropout_rate=dropout_rate
        self.encoder=Encoder(input_channels,num_layers,base_num,dropout_rate)
        self.mode=mode
        self.num_classes=num_classes
        if 'spatial' in mode or 'uncertainty' in mode:
            self.rank=rank
            self.num_classes=num_classes
            self.epsilon=epsilon
            self.diagonal=diagonal
            self.mean_l=nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=(1,1))
            self.log_cov_diag_l=nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=(1,1))
            self.cov_factor_l=nn.Conv2d(in_channels=base_num,out_channels=num_classes*rank,kernel_size=(1,1))
            self.decoder=Decoder(num_classes,num_layers,base_num,include_last=False)
        elif 'dirichlet' in mode:
            self.decoder=Decoder(num_classes,num_layers,base_num,include_last=True,dropout_rate=dropout_rate)
        else:
            self.decoder=Decoder(num_classes,num_layers,base_num,include_last=True,dropout_rate=dropout_rate)

    def forward(self,x):
        x,b=self.encoder(x)
        x=self.decoder(x,b)
        if self.mode=='softmax':
            x=torch.sigmoid(x)
            return x
        elif self.mode=='dirichlet':
            evidence=torch.exp(torch.clamp(x,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            return prob,uncertainty,S,alpha,evidence
        elif self.mode=='spatial':
            logits=F.relu(x)
            batch_size=logits.shape[0]
            event_shape=(self.num_classes,)+logits.shape[2:]

            mean=self.mean_l(logits)
            cov_diag=self.log_cov_diag_l(logits).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))

            cov_factor=self.cov_factor_l(logits)
            cov_factor=cov_factor.view((batch_size,self.rank,self.num_classes,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)

            # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI

            cov_diag=cov_diag+self.epsilon

            if self.diagonal:
                base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            else:
                try:
                    base_distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
                except:
                    print('Covariance became not invertible using independent normals for this batch!')
                    base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            distribution=ReshapedDistribution(base_distribution,event_shape)

            shape=(batch_size,)+event_shape
            logit_mean=mean.view(shape)
            cov_diag_view=cov_diag.view(shape).detach()
            cov_factor_view=cov_factor.transpose(2,1).view((batch_size,self.num_classes*self.rank)+event_shape[1:]).detach()

            output_dict={'logit_mean':logit_mean.detach(),
                         'cov_diag':cov_diag_view,
                         'cov_factor':cov_factor_view,
                         'distribution':distribution}

            return logit_mean,output_dict

        elif self.mode=='uncertainty':
            logits=F.relu(x)
            batch_size=logits.shape[0]
            event_shape=(self.num_classes,)+logits.shape[2:]

            mean=self.mean_l(logits)
            cov_diag=self.log_cov_diag_l(logits).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))

            cov_factor=self.cov_factor_l(logits)
            cov_factor=cov_factor.view((batch_size,self.rank,self.num_classes,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)

            # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI

            cov_diag=cov_diag+self.epsilon

            if self.diagonal:
                base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            else:
                try:
                    base_distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
                except:
                    print('Covariance became not invertible using independent normals for this batch!')
                    base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            distribution=ReshapedDistribution(base_distribution,event_shape)

            shape=(batch_size,)+event_shape
            logit_mean=mean.view(shape)
            cov_diag_view=cov_diag.view(shape).detach()
            cov_factor_view=cov_factor.transpose(2,1).view((batch_size,self.num_classes*self.rank)+event_shape[1:]).detach()
            evidence=torch.clamp(logit_mean,0.000001,1000)
            #evidence=torch.exp(torch.clamp(logit_mean,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            output_dict={'logit_mean':logit_mean.detach(),
                         'cov_diag':cov_diag_view,
                         'cov_factor':cov_factor_view,
                         'distribution':distribution}

            return prob,uncertainty,S,alpha,evidence,output_dict



class ProbBlock(nn.Module):
    def __init__(self,in_channels,out_channels,rank,epsilon=0.0001):
        super(ProbBlock,self).__init__()
        self.out_channels=out_channels
        if self.out_channels!=0:
            self.mean_l=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1))
            self.log_cov_diag_l=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1))
            self.cov_factor_l=nn.Conv2d(in_channels=in_channels,out_channels=out_channels*rank,kernel_size=(1,1))
            self.upsamp=nn.UpsamplingBilinear2d(scale_factor=2)
            self.epsilon=epsilon
            self.rank=rank
    def forward(self,x):
        if self.out_channels!=0:
            batch_size=x.shape[0]
            event_shape=(self.out_channels,)+x.shape[2:]
            mean=self.mean_l(x)
            cov_diag=torch.exp(self.log_cov_diag_l(x))
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))
            cov_factor=self.cov_factor_l(x)
            cov_factor=cov_factor.view((batch_size,self.rank,self.out_channels,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)
            cov_diag=cov_diag+self.epsilon
            distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
            distribution=ReshapedDistribution(distribution,event_shape)
            out=distribution.rsample((1,))
            out=out.squeeze(0)
            out=self.upsamp(out)
            return out,distribution
        else:
            return None,None
    def mean(self,x):
        if self.out_channels!=0:
            batch_size=x.shape[0]
            event_shape=(self.out_channels,)+x.shape[2:]
            mean=self.mean_l(x)
            cov_diag=self.log_cov_diag_l(x).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))
            cov_factor=self.cov_factor_l(x)
            cov_factor=cov_factor.view((batch_size,self.rank,self.out_channels,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)
            distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
            distribution=ReshapedDistribution(distribution,event_shape)
            out=distribution.mean
            out=self.upsamp(out)
            return out
        else:
            return None





class ProbDecoder(nn.Module):
    """Decoder for UNet"""
    def __init__(self,base_nums,prob_blocks,input_channels,num_classes,rank=10,include_last=True):
        super(ProbDecoder,self).__init__()
        self.include_last=include_last
        self.conv=[]
        self.prob=[]
        self.num_layers=len(base_nums)
        for i in range(len(base_nums)):
            if i!=0:
                self.conv.append(ProbDeconvBlock(base_nums[i-1],prob_blocks[i],base_nums[i]))
            else:
                self.conv.append(ProbDeconvBlock(input_channels,prob_blocks[i],base_nums[i]))

        if include_last:
            self.conv.append(nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=1,stride=1,padding=0))
        self.conv=nn.Sequential(*self.conv)
        for j in range(len(prob_blocks)):
            if j!=0:
                self.prob.append(ProbBlock(base_nums[j-1],prob_blocks[j],rank=rank))
            else:
                self.prob.append(ProbBlock(input_channels,prob_blocks[j],rank=rank))
        self.prob=nn.Sequential(*self.prob)
    def forward(self,x,b):
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                prob,_=self.prob[i](x)
                x=self.conv[i](x,b[i],prob)
            else:
                if self.include_last:
                    x=self.conv[i](x)
        return x
    def mean(self,x,b):
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                prob=self.prob[i].mean(x)
                x=self.conv[i](x,b[i],prob)
            else:
                if self.include_last:
                    x=self.conv[i](x)
        return x



class ProbDecoderPost(nn.Module):
    """Decoder for UNet"""
    def __init__(self,base_nums,prob_blocks,input_channels,rank=10):
        super(ProbDecoderPost,self).__init__()
        self.conv=[]
        self.prob=[]
        self.num_layers=len(base_nums)
        for i in range(len(base_nums)):
            if i!=0:
                self.conv.append(ProbDeconvBlock(base_nums[i-1],prob_blocks[i],base_nums[i]))
            else:
                self.conv.append(ProbDeconvBlock(input_channels,prob_blocks[i],base_nums[i]))

        self.conv=nn.Sequential(*self.conv)
        for j in range(len(prob_blocks)):
            if j!=0:
                self.prob.append(ProbBlock(base_nums[j-1],prob_blocks[j],rank=rank))
            else:
                self.prob.append(ProbBlock(input_channels,prob_blocks[j],rank=rank))
        self.prob=nn.Sequential(*self.prob)
    def forward(self,x,b):
        distributions=[]
        for i in range(self.num_layers):
            prob,d=self.prob[i](x)
            distributions.append(d)
            x=self.conv[i](x,b[i],prob)
        return x,distributions

class ProbDecoderPrior(nn.Module):
    """Decoder for UNet"""
    def __init__(self,base_nums,prob_blocks,input_channels,num_classes,rank=10):
        super(ProbDecoderPrior,self).__init__()
        self.conv=[]
        self.prob=[]
        self.num_layers=len(base_nums)
        for i in range(len(base_nums)):
            if i!=0:
                self.conv.append(ProbDeconvBlock(base_nums[i-1],prob_blocks[i],base_nums[i]))
            else:
                self.conv.append(ProbDeconvBlock(input_channels,prob_blocks[i],base_nums[i]))

        self.conv=nn.Sequential(*self.conv)
        for j in range(len(prob_blocks)):
            if j!=0:
                self.prob.append(ProbBlock(base_nums[j-1],prob_blocks[j],rank=rank))
            else:
                self.prob.append(ProbBlock(input_channels,prob_blocks[j],rank=rank))
        self.prob=nn.Sequential(*self.prob)
        self.upsamp=nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_final=nn.Conv2d(in_channels=base_nums[-1],out_channels=num_classes,kernel_size=(1,1))
    def forward(self,x,b,ds):
        distributions=[]
        for i in range(self.num_layers):
            _,d=self.prob[i](x)
            distributions.append(d)
            if ds[i] is None:
                prob=None
            else:
                out=ds[i].rsample((1,))
                out=out.squeeze(0)
                prob=self.upsamp(out)
            x=self.conv[i](x,b[i],prob)
        x=self.conv_final(x)
        return x,distributions
    def sample(self,x,b):
        for i in range(self.num_layers):
            prob=self.prob[i](x)
            x=self.conv[i](x,b[i],prob)
        x=self.conv_final(x)
        return x
    def mean(self,x,b):
        for i in range(self.num_layers):
            prob=self.prob[i].mean(x)
            x=self.conv[i](x,b[i],prob)
        x=self.conv_final(x)
        return x

class DeepProbEncoder(nn.Module):
    """Encoder for both UNet and UNet transformer"""
    def __init__(self,input_channels,base_nums):
        super(DeepProbEncoder,self).__init__()
        self.conv=[]
        self.num_layers=len(base_nums)
        for i in range(len(base_nums)):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_nums[0],True))
            else:
                self.conv.append(ConvBlock(base_nums[i-1],base_nums[i],(i!=self.num_layers-1)))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b


class ProbUNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,prob_blocks,mode='softmax',base_num=64,rank=10):
        super(ProbUNet,self).__init__()
        self.encoder=Encoder(input_channels,num_layers,base_num)
        self.mode=mode
        self.num_classes=num_classes
        self.decoder=ProbDecoder(num_classes,num_layers,base_num,prob_blocks,rank=rank,include_last=True)
    def forward(self,x):
        x,b=self.encoder(x)
        x=self.decoder(x,b)
        if self.mode=='dirichlet':
            evidence=torch.exp(torch.clamp(x,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            return prob,uncertainty,S,alpha,evidence
        elif self.mode=='softmax':
            return x
        elif self.mode=='uncertainty':
            logits=F.relu(x)
            batch_size=logits.shape[0]
            event_shape=(self.num_classes,)+logits.shape[2:]

            mean=self.mean_l(logits)
            cov_diag=self.log_cov_diag_l(logits).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))

            cov_factor=self.cov_factor_l(logits)
            cov_factor=cov_factor.view((batch_size,self.rank,self.num_classes,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)

            # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI

            cov_diag=cov_diag+self.epsilon

            if self.diagonal:
                base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            else:
                try:
                    base_distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
                except:
                    print('Covariance became not invertible using independent normals for this batch!')
                    base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            distribution=ReshapedDistribution(base_distribution,event_shape)

            shape=(batch_size,)+event_shape
            logit_mean=mean.view(shape)
            cov_diag_view=cov_diag.view(shape).detach()
            cov_factor_view=cov_factor.transpose(2,1).view((batch_size,self.num_classes*self.rank)+event_shape[1:]).detach()
            evidence=torch.clamp(logit_mean,0.000001,1000)
            #evidence=torch.exp(torch.clamp(logit_mean,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            output_dict={'logit_mean':logit_mean.detach(),
                         'cov_diag':cov_diag_view,
                         'cov_factor':cov_factor_view,
                         'distribution':distribution}

            return prob,uncertainty,S,alpha,evidence,output_dict
    def mean(self,x):
        x,b=self.encoder(x)
        x=self.decoder.mean(x,b)
        if self.mode=='dirichlet':
            evidence=torch.exp(torch.clamp(x,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            return prob,uncertainty,S,alpha,evidence
        elif self.mode=='softmax':
            return x
        elif self.mode=='uncertainty':
            logits=F.relu(x)
            batch_size=logits.shape[0]
            event_shape=(self.num_classes,)+logits.shape[2:]

            mean=self.mean_l(logits)
            cov_diag=self.log_cov_diag_l(logits).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))

            cov_factor=self.cov_factor_l(logits)
            cov_factor=cov_factor.view((batch_size,self.rank,self.num_classes,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)

            # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI

            cov_diag=cov_diag+self.epsilon

            if self.diagonal:
                base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            else:
                try:
                    base_distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
                except:
                    print('Covariance became not invertible using independent normals for this batch!')
                    base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            distribution=ReshapedDistribution(base_distribution,event_shape)

            shape=(batch_size,)+event_shape
            logit_mean=mean.view(shape)
            cov_diag_view=cov_diag.view(shape).detach()
            cov_factor_view=cov_factor.transpose(2,1).view((batch_size,self.num_classes*self.rank)+event_shape[1:]).detach()
            evidence=torch.clamp(logit_mean,0.000001,1000)
            #evidence=torch.exp(torch.clamp(logit_mean,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            output_dict={'logit_mean':logit_mean.detach(),
                         'cov_diag':cov_diag_view,
                         'cov_factor':cov_factor_view,
                         'distribution':distribution}

            return prob,uncertainty,S,alpha,evidence,output_dict






class DeepProbUNet(nn.Module):
    def __init__(self,input_channels,num_classes,base_nums,prob_blocks,mode='softmax',rank=40):
        super(DeepProbUNet,self).__init__()
        self.encoder_prior=DeepProbEncoder(input_channels,base_nums)
        self.encoder_post=DeepProbEncoder(input_channels+num_classes,base_nums)
        self.mode=mode
        self.num_classes=num_classes
        temp=base_nums[-1]
        base_nums=base_nums[:-1]
        base_nums=base_nums[::-1]
        self.decoder_post=ProbDecoderPost(base_nums,prob_blocks,temp,rank)
        self.decoder_prior=ProbDecoderPrior(base_nums,prob_blocks,temp,num_classes,rank)
    def forward(self,x,y):
        x_post,b_post=self.encoder_post(torch.cat((x,y),dim=1))
        x_prior,b_prior=self.encoder_prior(x)
        x_post,distributions_post=self.decoder_post(x_post,b_post)
        x,distributions_prior=self.decoder_prior(x_prior,b_prior,distributions_post)
        if self.mode=='dirichlet':
            evidence=torch.exp(torch.clamp(x,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            return prob,uncertainty,S,alpha,evidence,distributions_post,distributions_prior
        elif self.mode=='softmax':
            return x,distributions_post,distributions_prior
        elif self.mode=='uncertainty':
            logits=F.relu(x)
            batch_size=logits.shape[0]
            event_shape=(self.num_classes,)+logits.shape[2:]

            mean=self.mean_l(logits)
            cov_diag=self.log_cov_diag_l(logits).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))

            cov_factor=self.cov_factor_l(logits)
            cov_factor=cov_factor.view((batch_size,self.rank,self.num_classes,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)

            # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI

            cov_diag=cov_diag+self.epsilon

            if self.diagonal:
                base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            else:
                try:
                    base_distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
                except:
                    print('Covariance became not invertible using independent normals for this batch!')
                    base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            distribution=ReshapedDistribution(base_distribution,event_shape)

            shape=(batch_size,)+event_shape
            logit_mean=mean.view(shape)
            cov_diag_view=cov_diag.view(shape).detach()
            cov_factor_view=cov_factor.transpose(2,1).view((batch_size,self.num_classes*self.rank)+event_shape[1:]).detach()
            evidence=torch.clamp(logit_mean,0.000001,1000)
            #evidence=torch.exp(torch.clamp(logit_mean,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            output_dict={'logit_mean':logit_mean.detach(),
                         'cov_diag':cov_diag_view,
                         'cov_factor':cov_factor_view,
                         'distribution':distribution}
            return prob,uncertainty,S,alpha,evidence,output_dict,distributions_post,distributions_prior
    def sample(self,x):
        x_prior,b_prior=self.encoder_prior(x)
        x,distributions_prior=self.decoder_prior.sample(x_prior,b_prior)
        if self.mode=='dirichlet':
            evidence=torch.exp(torch.clamp(x,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            return prob,uncertainty,S,alpha,evidence,distributions_prior
        elif self.mode=='softmax':
            return x,distributions_prior
        elif self.mode=='uncertainty':
            logits=F.relu(x)
            batch_size=logits.shape[0]
            event_shape=(self.num_classes,)+logits.shape[2:]

            mean=self.mean_l(logits)
            cov_diag=self.log_cov_diag_l(logits).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))

            cov_factor=self.cov_factor_l(logits)
            cov_factor=cov_factor.view((batch_size,self.rank,self.num_classes,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)

            # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI

            cov_diag=cov_diag+self.epsilon

            if self.diagonal:
                base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            else:
                try:
                    base_distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
                except:
                    print('Covariance became not invertible using independent normals for this batch!')
                    base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            distribution=ReshapedDistribution(base_distribution,event_shape)

            shape=(batch_size,)+event_shape
            logit_mean=mean.view(shape)
            cov_diag_view=cov_diag.view(shape).detach()
            cov_factor_view=cov_factor.transpose(2,1).view((batch_size,self.num_classes*self.rank)+event_shape[1:]).detach()
            evidence=torch.clamp(logit_mean,0.000001,1000)
            #evidence=torch.exp(torch.clamp(logit_mean,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            output_dict={'logit_mean':logit_mean.detach(),
                         'cov_diag':cov_diag_view,
                         'cov_factor':cov_factor_view,
                         'distribution':distribution}
            return prob,uncertainty,S,alpha,evidence,output_dict,distributions_prior

    def mean(self,x):
        x_prior,b_prior=self.encoder_prior(x)
        x,distributions_prior=self.decoder_prior.mean(x_prior,b_prior)
        if self.mode=='dirichlet':
            evidence=torch.exp(torch.clamp(x,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            return prob,uncertainty,S,alpha,evidence,distributions_prior
        elif self.mode=='softmax':
            return x,distributions_prior
        elif self.mode=='uncertainty':
            logits=F.relu(x)
            batch_size=logits.shape[0]
            event_shape=(self.num_classes,)+logits.shape[2:]

            mean=self.mean_l(logits)
            cov_diag=self.log_cov_diag_l(logits).exp()+self.epsilon
            mean=mean.view((batch_size,-1))
            cov_diag=cov_diag.view((batch_size,-1))

            cov_factor=self.cov_factor_l(logits)
            cov_factor=cov_factor.view((batch_size,self.rank,self.num_classes,-1))
            cov_factor=cov_factor.flatten(2,3)
            cov_factor=cov_factor.transpose(1,2)

            # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI

            cov_diag=cov_diag+self.epsilon

            if self.diagonal:
                base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            else:
                try:
                    base_distribution=td.LowRankMultivariateNormal(loc=mean,cov_factor=cov_factor,cov_diag=cov_diag)
                except:
                    print('Covariance became not invertible using independent normals for this batch!')
                    base_distribution=td.Independent(td.Normal(loc=mean,scale=torch.sqrt(cov_diag)),1)
            distribution=ReshapedDistribution(base_distribution,event_shape)

            shape=(batch_size,)+event_shape
            logit_mean=mean.view(shape)
            cov_diag_view=cov_diag.view(shape).detach()
            cov_factor_view=cov_factor.transpose(2,1).view((batch_size,self.num_classes*self.rank)+event_shape[1:]).detach()
            evidence=torch.clamp(logit_mean,0.000001,1000)
            #evidence=torch.exp(torch.clamp(logit_mean,-10,10))
            alpha=evidence+1
            S=torch.sum(alpha,dim=1,keepdim=True)
            uncertainty=self.num_classes/S
            prob=alpha/S
            output_dict={'logit_mean':logit_mean.detach(),
                         'cov_diag':cov_diag_view,
                         'cov_factor':cov_factor_view,
                         'distribution':distribution}
            return prob,uncertainty,S,alpha,evidence,output_dict,distributions_prior







def PositionalEncoding2d(d_model,height,width):
    """
    Generate a 2D positional Encoding

    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    #https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    if d_model%4!=0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    height=int(height)
    width=int(width)
    pe = torch.zeros((d_model,height,width)).to(device)
    # Each dimension use half of d_model
    d_model=int(d_model/2)
    div_term=torch.exp(torch.arange(0.,d_model,2)*(-(math.log(10000.0)/d_model)))
    pos_w=torch.arange(0.,width).unsqueeze(1)
    pos_h=torch.arange(0.,height).unsqueeze(1)
    pe[0:d_model:2,:,:]=torch.sin(pos_w * div_term).transpose(0,1).unsqueeze(1).repeat(1,height,1)
    pe[1:d_model:2,:,:]=torch.cos(pos_w*div_term).transpose(0,1).unsqueeze(1).repeat(1,height,1)
    pe[d_model::2,:,:]=torch.sin(pos_h * div_term).transpose(0,1).unsqueeze(2).repeat(1,1,width)
    pe[d_model+1::2,:,:] = torch.cos(pos_h*div_term).transpose(0,1).unsqueeze(2).repeat(1,1,width)
    pe=torch.reshape(pe,(1,d_model*2,height,width))
    return pe

class MHSA(nn.Module):
    """Multihead self attetion module with positional encoding"""
    def __init__(self,in_dim,h,w):
        super(MHSA,self).__init__()
        self.pe=PositionalEncoding2d(in_dim,h,w)
        self.query_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.key_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.value_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        #self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize,C,height,width=x.size()
        x=self.pe+x
        proj_query=self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)  # B X (WH) X C
        proj_key=self.key_conv(x).view(m_batchsize,-1,width*height)  # B X C X (WH)
        energy=torch.bmm(proj_query,proj_key)
        attention=self.softmax(energy)  # B X (WH) X (WH)
        proj_value=self.value_conv(x).view(m_batchsize,-1,width*height)  # B X C X (WH)
        out=torch.bmm(proj_value,attention.permute(0,2,1))
        out=out.view(m_batchsize,C,height,width)
        #out=self.gamma*out+x
        return out,attention

class MHCA(nn.Module):
    """multihead cross attention"""
    def __init__(self,d1,h1,w1,d2,h2,w2):
        super(MHCA,self).__init__()
        self.d1=d1
        self.h1=h1
        self.w1=w1
        self.d2=d2
        self.h2=h2
        self.w2=w2
        self.pe1=PositionalEncoding2d(d1,h1,w1)
        self.pe2=PositionalEncoding2d(d2,h2,w2)
        self.query_conv=[]
        self.key_conv=[]
        self.value_conv=[]
        self.query_conv.append(nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False))
        #self.query_conv.append(nn.BatchNorm2d(d1,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        #self.query_conv.append(nn.ReLU())
        self.query_conv=nn.Sequential(*self.query_conv)
        self.key_conv.append(nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False))
        #self.key_conv.append(nn.BatchNorm2d(d1,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        #self.key_conv.append(nn.ReLU())
        self.key_conv=nn.Sequential(*self.key_conv)
        self.value_conv.append(nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1,stride=2,bias=False))
        #self.value_conv.append(nn.BatchNorm2d(d2,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        #self.value_conv.append(nn.ReLU())

        # to be fixed
        #self.value_conv.append(nn.MaxPool2d(2,stride=2,dilation=(1,1)))

        self.value_conv=nn.Sequential(*self.value_conv)
        #self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1))
        self.conv.append(nn.BatchNorm2d(d2,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.Sigmoid())
        self.conv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x,b):
        batch_size=x.size(0)
        #print(x.size(),b.size(),self.pe1.size(),self.pe2.size())
        x=self.pe1+x
        b=self.pe2+b
        #print(x.size(),b.size())
        q=self.query_conv(x).view(batch_size,-1,self.w1*self.h1).permute(0,2,1)
        k=self.key_conv(x).view(batch_size,-1,self.w1*self.h1)
        #v=self.value_conv(b).view(batch_size,-1,self.w2*self.h2)
        v=self.value_conv(b).view(batch_size,-1,self.w1*self.h1).permute(0,2,1)
        #print(q.size(),k.size(),v.size())
        energy=torch.bmm(q,k)
        attention=self.softmax(energy)
        #print(v.size(),attention.size(),energy.size())
        out=torch.bmm(attention,v)
        out=out.view(batch_size,self.d2,self.h1,self.w1)
        #print(out.size(),x.size())
        #out=self.gamma*out+b
        out=self.conv(out)
        out=out*b
        return out,attention,x


class SelfAxialAttention(nn.Module):
    def __init__(self,channel,in_dim,h,w):
        super(SelfAxialAttention,self).__init__()
        self.channel=channel
        self.h=h
        self.w=w
        self.in_dim=in_dim
        if channel=='h':
            self.pe=PositionalEncoding2d(in_dim,h,1)
        elif channel=='w':
            self.pe=PositionalEncoding2d(in_dim,1,w)
        self.query_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.key_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.value_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1,bias=False)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize,C,height,width=x.size()
        if self.channel=='h':
            x=x.permute(0,3,1,2).contiguous().view(-1,C,height,1)
        elif self.channel=='w':
            x=x.permute(0,2,1,3).contiguous().view(-1,C,1,width)
        x=x+self.pe
        m_batchsize_,C_,height_,width_=x.size()
        proj_query=self.query_conv(x).view(m_batchsize_,-1,width_*height_).permute(0,2,1)  # B X (WH) X C
        proj_key=self.key_conv(x).view(m_batchsize_,-1,width_*height_)  # B X C X (WH)
        proj_value=self.value_conv(x).view(m_batchsize_,-1,width_*height_)  # B X C X (WH)
        energy=torch.bmm(proj_query,proj_key)
        attention=self.softmax(energy)  # B X (WH) X (WH)
        out=torch.bmm(proj_value,attention.permute(0,2,1))
        out=out.view(m_batchsize_,C_,height_,width_)
        if self.channel=='h':
            out=out.squeeze().view(m_batchsize,width,C,height).permute(0,2,3,1)
        elif self.channel=='w':
            out=out.squeeze().view(m_batchsize,height,C,width).permute(0,2,1,3)
        return out,attention

class SelfAxialAttentionBoth(nn.Module):
    def __init__(self,in_dim,h,w):
        super(SelfAxialAttentionBoth,self).__init__()
        self.attention1=SelfAxialAttention('h',in_dim,h,w)
        self.attention2=SelfAxialAttention('w',in_dim,h,w)
    def forward(self,x):
        attention=[]
        x,temp_attention=self.attention1(x)
        attention.append(temp_attention)
        x,temp_attention=self.attention2(x)
        attention.append(temp_attention)
        return x,attention

class CrossAxialAttention(nn.Module):
    def __init__(self,d1,h1,w1,d2,h2,w2):
        super(CrossAxialAttention,self).__init__()
        self.d1=d1
        self.h1=h1
        self.w1=w1
        self.d2=d2
        self.h2=h2//2
        self.w2=w2//2
        self.pe1_h=PositionalEncoding2d(d1,h1,1)
        self.pe2_h=PositionalEncoding2d(d2,self.h2,1)
        self.pe1_w=PositionalEncoding2d(d1,1,h1)
        self.pe2_w=PositionalEncoding2d(d2,1,self.h2)
        self.query_conv_h=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.key_conv_h=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.value_conv_h=nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1,bias=False)
        self.query_conv_w=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.key_conv_w=nn.Conv2d(in_channels=d1,out_channels=d1,kernel_size=1,bias=False)
        self.value_conv_w=nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1,bias=False)
        self.softmax_h=nn.Softmax(dim=-1)
        self.softmax_w=nn.Softmax(dim=-1)
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=d2,out_channels=d2,kernel_size=1))
        self.conv.append(nn.BatchNorm2d(d2,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True))
        self.conv.append(nn.Sigmoid())
        self.conv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv=nn.Sequential(*self.conv)
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=2)

    def forward(self,x,b):
        attentions=[]
        b_=self.pool(b)
        batch_size1,C1,height1,width1=x.size()
        batch_size2,C2,height2,width2=b_.size()
        x=x.permute(0,3,1,2).contiguous().view(-1,C1,height1,1)
        b_=b_.permute(0,3,1,2).contiguous().view(-1,C2,height2,1)
        x=self.pe1_h+x
        b_=self.pe2_h+b_
        batch_size1_,C1_,height1_,width1_=x.size()
        batch_size2_,C2_,height2_,width2_=b_.size()
        q=self.query_conv_h(x).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        k=self.key_conv_h(x).view(batch_size1_,-1,height1_*width1_)
        v=self.value_conv_h(b_).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        energy=torch.bmm(q,k)
        attention=self.softmax_h(energy)
        attentions.append(attention)
        out=torch.bmm(attention,v)
        out=out.view(batch_size1_,C2_,height1_,width1_)
        out=out.squeeze().view(batch_size1,width1,C2,height1).permute(0,2,3,1)
        x=x.squeeze().view(batch_size1,width1,C1,height1).permute(0,2,3,1)
        b_=out
        x=x.permute(0,2,1,3).contiguous().view(-1,C1,1,width1)
        b_=b_.permute(0,2,1,3).contiguous().view(-1,C2,1,width2)
        x=self.pe1_w+x
        b_=self.pe2_w+b_
        batch_size1_,C1_,height1_,width1_=x.size()
        batch_size2_,C2_,height2_,width2_=b_.size()
        q=self.query_conv_w(x).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        k=self.key_conv_w(x).view(batch_size1_,-1,height1_*width1_)
        v=self.value_conv_w(b_).view(batch_size1_,-1,height1_*width1_).permute(0,2,1)
        energy=torch.bmm(q,k)
        attention=self.softmax_w(energy)
        attentions.append(attentions)
        out=torch.bmm(attention,v)
        out=out.view(batch_size1_,C2_,height1_,width1_)
        out=out.squeeze().view(batch_size1,height1,C2,width1).permute(0,2,1,3)
        x=x.squeeze().view(batch_size1,height1,C1,width1).permute(0,2,1,3)
        out=self.conv(out)
        out=out*b
        return out,attentions,x



class DeconvBlockTransformer(nn.Module):
    """DeconvBlock for transformer UNet"""
    def __init__(self,d1,h1,w1,d2,h2,w2,block):
        super(DeconvBlockTransformer,self).__init__()
        self.upconv=[]
        if block=='attention':
            self.MHCA=MHCA(d1,h1,w1,d2,h2,w2)
        elif block=='axial':
            self.cross_axial_attention=CrossAxialAttention(d1,h1,w1,d2,h2,w2)
        self.block=block
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=d1,out_channels=d2,kernel_size=3,stride=1,padding=1))
        self.upconv.append(nn.ReLU())
        self.conv=ConvBlock(d2*2,d2,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        if self.block=='attention':
            b,attention,x=self.MHCA(x,b)
            attention=[attention]
        elif self.block=='axial':
            b,attention,x=self.cross_axial_attention(x,b)
        else:
            attention=[None]
        x=self.upconv(x)
        x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x,attention


class DecoderTransformer(nn.Module):
    """Decoder for transformer UNet"""
    def __init__(self,num_classses,num_layers,h,w,blocks,base_num):
        super(DecoderTransformer,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        k=0
        for i in range(num_layers-1,0,-1):
            kk=k+1
            self.conv.append(DeconvBlockTransformer(base_num*(2**i),h*(2**k),w*(2**k),base_num*(2**(i-1)),h*(2**kk),w*(2**kk),blocks[k]))
            k=kk
        self.conv.append(nn.Conv2d(in_channels=base_num,out_channels=num_classses,kernel_size=1,stride=1,padding=0))
        self.conv=nn.Sequential(*self.conv)
    def forward(self,x,b):
        attention=[]
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                x,attention_temp=self.conv[i](x,b[i])
                attention+=attention_temp
            else:
                x=self.conv[i](x)
        return x,attention



class UNetTransformer(nn.Module):
    """UNet transformer"""
    def __init__(self,input_channels,num_classes,num_layers,input_h,input_w,blocks,base_num=64):
        super(UNetTransformer,self).__init__()
        self.h=int(input_h/(2**(num_layers-1)))
        self.w=int(input_w/(2**(num_layers-1)))
        self.encoder=Encoder(input_channels,num_layers,base_num)
        self.decoder=DecoderTransformer(num_classes,num_layers,self.h,self.w,blocks[1:],base_num)
        self.blocks=blocks
        if blocks[0]=='attention':
            self.MHSA=MHSA(base_num*(2**(num_layers-1)),self.h,self.w)
        elif blocks[0]=='axial':
            self.self_axial_attention=SelfAxialAttentionBoth(base_num*(2**(num_layers-1)),self.h,self.w)
    def forward(self,x):
        attention=[]
        x,b=self.encoder(x)
        if self.blocks[0]=='attention':
            x,temp_attention=self.MHSA(x)
            temp_attention=[temp_attention]
        elif self.blocks[0]=='axial':
            x,temp_attention=self.self_axial_attention(x)
        else:
            temp_attention=[None]
        attention+=temp_attention
        x,temp_attention=self.decoder(x,b)
        attention=attention+temp_attention
        x=torch.sigmoid(x)
        #return x,attention
        return x


####https://github.com/asyml/vision-transformer-pytorch
class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim).to(device)
        self.fc2 = nn.Linear(mlp_dim, out_dim).to(device)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class SelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(SelfAttention,self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim).to(device)
        self.o_proj = nn.Linear(embed_dim, embed_dim).to(device)

        self._reset_parameters()


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o

class TransBlock(nn.Module):
    def __init__(self,in_dim,emb_dim,mlp_dim,num_heads,dropout_rate=0.1):
        super(TransBlock,self).__init__()
        in_dim=int(in_dim)
        mlp_dim=int(mlp_dim)
        num_heads=int(num_heads)
        self.norm1=nn.LayerNorm(in_dim).to(device)
        self.attn=SelfAttention(in_dim,emb_dim,num_heads=num_heads)
        if dropout_rate>0:
            self.dropout=nn.Dropout(dropout_rate)
        else:
            self.dropout=None
        self.norm2=nn.LayerNorm(in_dim).to(device)
        self.mlp=MlpBlock(in_dim,mlp_dim,in_dim,dropout_rate)

    def forward(self,x):
        residual=x
        out=self.norm1(x)
        out=self.attn(out)
        if self.dropout:
            out=self.dropout(out)
        out+=residual
        residual=out
        out=self.norm2(out)
        out=self.mlp(out)
        out+=residual
        return out

class TransBlocks(nn.Module):
    def __init__(self,num_blocks,in_dim,emb_dim,mlp_dim,num_heads,dropout_rate=0.1):
        super(TransBlocks,self).__init__()
        self.blocks=[]
        for i in range(num_blocks):
            self.blocks.append(TransBlock(in_dim,emb_dim,mlp_dim,num_heads,dropout_rate))
        self.blocks=nn.Sequential(*self.blocks)
    def forward(self,x):
        x_=x
        b,c,h,w=x.size()
        x=x.view(b,-1,c)
        x=self.blocks(x)
        x=x.view(b,c,h,w)
        x=x+x_
        return x





class LocalUTrans(nn.Module):
    def __init__(self,in_channels,num_heads,num_blocks,base_num=8,dropout_rate=0.1):
        super(LocalUTrans,self).__init__()
        self.conv=ConvBlock(input_channels=in_channels,output_channels=4,max_pool=False)
        self.encoder=Encoder(4,4,base_num)
        self.decoder=Decoder(base_num,4,base_num)
        self.transformers=[]
        self.base_num=base_num
        for i in range(4):
            self.transformers.append(TransBlocks(num_blocks,base_num*(2**i),base_num*(2**i),base_num*(2**i)/8,num_heads,dropout_rate))
        self.transformers=self.transformers[::-1]
        self.pos_emb=PositionalEncoding2d(4,256,256)

    def forward(self,x):
        x,_=self.conv(x)
        x=x+self.pos_emb
        x_temp=torch.zeros((x.size(0),self.base_num,256,256))
        for i in range(0,256,16):
            for j in range(0,256,16):
                x_,b=self.encoder(x[:,:,i:i+16,j:j+16])
                for k in range(len(self.transformers)):
                    if k==0:
                        x_=self.transformers[k](x_)
                    else:
                        b[k-1]=self.transformers[k](b[k-1])
                x_=self.decoder(x_,b)
                x_temp[:,:,i:i+16,j:j+16]=x_
        return x_temp

class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ASPP,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv2=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=2,dilation=2)
        self.conv3=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=4,dilation=4)
        self.conv4=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=8,dilation=8)
        self.bnorm=nn.BatchNorm2d(4*out_channels,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
        self.relu=nn.ReLU()
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        x4=self.conv4(x)
        x=torch.cat((x1,x2,x3,x4),dim=1)
        x=self.bnorm(x)
        x=self.relu(x)
        return x



class GlobalUASPP(nn.Module):
    def __init__(self,in_channels,out_channels,base_num=8):
        super(GlobalUASPP,self).__init__()
        self.conv1_1=ASPP(in_channels,base_num)
        self.conv1_2=ASPP(base_num*4,base_num)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=(2,2),dilation=(1,1))
        self.conv2_1=ASPP(4*base_num,2*base_num)
        self.conv2_2=ASPP(8*base_num,2*base_num)
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=(2,2),dilation=(1,1))
        self.conv3_1=ASPP(8*base_num,4*base_num)
        self.conv3_2=ASPP(16*base_num,4*base_num)
        self.up2=nn.UpsamplingBilinear2d(scale_factor=2)
        self.upconv2=nn.Conv2d(in_channels=16*base_num,out_channels=8*base_num,kernel_size=3,stride=1,padding=1)
        self.deconv2_2=ASPP(16*base_num,2*base_num)
        self.deconv2_1=ASPP(8*base_num,2*base_num)
        self.up1=nn.UpsamplingBilinear2d(scale_factor=2)
        self.upconv1=nn.Conv2d(in_channels=8*base_num,out_channels=4*base_num,kernel_size=3,stride=1,padding=1)
        self.deconv1_2=ASPP(8*base_num,base_num)
        self.deconv1_1=ASPP(4*base_num,base_num)
        self.final_conv=nn.Conv2d(in_channels=4*base_num,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        x=self.conv1_1(x)
        x=self.conv1_2(x)
        b1=x
        x=self.maxpool1(x)
        x=self.conv2_1(x)
        x=self.conv2_2(x)
        b2=x
        x=self.maxpool2(x)
        x=self.conv3_1(x)
        x=self.conv3_2(x)
        x=self.up2(x)
        x=self.upconv2(x)
        x=torch.cat((x,b2),dim=1)
        x=self.deconv2_2(x)
        x=self.deconv2_1(x)
        x=self.up1(x)
        x=self.upconv1(x)
        x=torch.cat((x,b1),dim=1)
        x=self.deconv1_2(x)
        x=self.deconv1_1(x)
        x=self.final_conv(x)
        return x

class LocalGlobalTransformerASPPNet(nn.Module):
    def __init__(self,in_channels,num_classes,num_heads,num_blocks,local_base_num=16,global_base_num=16,dropout_rate=0.1):
        super(LocalGlobalTransformerASPPNet,self).__init__()
        self.local_net=LocalUTrans(in_channels,num_heads,num_blocks,local_base_num,dropout_rate)
        self.global_net=GlobalUASPP(in_channels,local_base_num,global_base_num)
        self.conv=nn.Conv2d(in_channels=local_base_num,out_channels=num_classes,kernel_size=1,stride=1,padding=0)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x_local=self.local_net(x)
        x_global=self.global_net(x)
        x_local=x_local.to(device)
        x_global=x_global.to(device)
        x=x_local+x_global
        x=self.conv(x)
        x=self.softmax(x)
        return x

class LocalGlobalTransformerUNet(nn.Module):
    def __init__(self,in_channels,num_classes,num_heads,num_blocks,local_base_num=16,global_base_num=64,dropout_rate=0.1):
        super(LocalGlobalTransformerUNet,self).__init__()
        self.local_net=LocalUTrans(in_channels,num_heads,num_blocks,local_base_num,dropout_rate)
        self.global_net=UNet(in_channels,global_base_num,5,global_base_num)
        self.conv=nn.Conv2d(in_channels=local_base_num,out_channels=num_classes,kernel_size=1,stride=1,padding=0)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x_local=self.local_net(x)
        x_global=self.global_net(x)
        x_local=x_local.to(device)
        x_global=x_global.to(device)
        x=x_local+x_global
        x=self.conv(x)
        x=self.softmax(x)
        return x


class FeaturePyramidAttention(nn.Module):
    def __init__(self, channels=2048):
        """
        Module of Feature Pyramid Attention
        :type channels: int
        """
        super(FeaturePyramidAttention, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)

        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)

        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_23 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_23 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv_final=nn.Conv2d(2*self.channels_cond, channels, kernel_size=1, bias=False)
    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """

        #Pooling branch for global pooling
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Branch 1 (two 7x7 convolutions)
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)

        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2 (two 5x5 convolutions)
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3 (two 3x3 convolutions)
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        ## To make the feature maps with mutiple scales, we firstly need to merge
        ##results from the three branches below, then multiplied with features maps
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x23_merge = self.relu(x2_2 + x3_upsample) #merging branch 2 and branch 3
        x23_upsample = self.relu(self.bn_upsample_23(self.conv_upsample_23(x23_merge)))
        x123_merge = self.relu(x1_2 + x23_upsample) #x123_merge contains the feature maps from the three branches
        x_integrated_with_mutiple_scales = x_master * self.relu\
        (self.bn_upsample_1(self.conv_upsample_1(x123_merge)))    # multiplied with feature maps after after a 1 × 1 convolution

        # integrate all the results (features maps,features maps integrated with mutiple scales,and global attention)
        out = self.relu(x + x_gpb+x_integrated_with_mutiple_scales)

        return out

class SAM(nn.Module):
    def __init__(self,input_channels=1,num_channels=64):
        super(SAM,self).__init__()
        self.q_conv=nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=(1,1))
        self.k_conv=nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=(1,1))
        self.v_conv=nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=(1,1))
        self.out_conv=nn.Conv2d(in_channels=num_channels,out_channels=input_channels,kernel_size=(1,1))
    def forward(self,x):
        q=self.q_conv(x).permute(0,2,3,1)
        k=self.k_conv(x).permute(0,2,3,1)
        v=self.v_conv(x).permute(0,2,3,1)
        temp_size=q.size(1)
        q=q.view(q.size(0),-1,q.size(3))
        k=k.view(k.size(0),-1,k.size(3))
        v=v.view(v.size(0),-1,v.size(3))
        temp=torch.bmm(q,k.permute(0,2,1))
        temp=F.softmax(temp,dim=-1)
        v=torch.bmm(temp,v)
        v=v.unsqueeze(-2)
        v=v.view(v.size(0),temp_size,temp_size,v.size(3))
        v=v.permute(0,3,1,2)
        v=self.out_conv(v)
        return v+x

class YK(nn.Module):
    def __init__(self,input_channels=1,num_classes=3,is_sam=True):
        super(YK,self).__init__()
        self.is_sam=is_sam
        if self.is_sam:
            self.SAM=SAM()
        self.resnet50=models.resnet50()
        self.resnet50.conv1=nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.resnet50.layer4[2].conv2=nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=2,dilation=2,bias=False)
        self.resnet50.layer4[0].downsample[0]=nn.Conv2d(1024,2048,kernel_size=(1,1),stride=(1,1))
        self.resnet50.layer4[0].conv2=nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.FPA1=FeaturePyramidAttention(256)
        self.FPA2=FeaturePyramidAttention(512)
        self.FPA3=FeaturePyramidAttention(1024)
        self.FPA4=FeaturePyramidAttention(2048)
        self.dropout1=nn.Dropout2d(p=0.2)
        self.dropout2=nn.Dropout2d(p=0.2)
        self.dropout3=nn.Dropout2d(p=0.2)
        self.dropout4=nn.Dropout2d(p=0.2)
        self.decoder=[]
        self.decoder.append(ConvBlock(2048*2-256,2048*2-256,False,return_single=True))
        self.decoder.append(nn.UpsamplingBilinear2d(scale_factor=4))
        self.decoder.append(nn.Conv2d(2048*2-256,num_classes,padding=0,kernel_size=(1,1),stride=(1,1),bias=False))
        self.decoder=nn.Sequential(*self.decoder)
        self.softmax=nn.Softmax(dim=1)
        self.upsamp4=nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsamp3=nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsamp2=nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self,x):
        if self.is_sam:
            x=self.SAM(x)
        x=self.resnet50.conv1(x)
        x=self.resnet50.bn1(x)
        x=self.resnet50.relu(x)
        x=self.resnet50.maxpool(x)
        x1=self.resnet50.layer1(x)
        x1=self.dropout1(x1)
        x2=self.resnet50.layer2(x1)
        x2=self.dropout2(x2)
        x3=self.resnet50.layer3(x2)
        x3=self.dropout3(x3)
        x4=self.resnet50.layer4(x3)
        x4=self.dropout4(x4)
        x1=self.FPA1(x1)
        x2=self.FPA2(x2)
        x2=self.upsamp2(x2)
        x3=self.FPA3(x3)
        x3=self.upsamp3(x3)
        x4=self.FPA4(x4)
        x4=self.upsamp4(x4)
        x=torch.cat((x1,x2,x3,x4),dim=1)
        x=self.decoder(x)
        #x=self.softmax(x)
        return x


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class ResNet50Pos(nn.Module):
    def __init__(self,num_conv_lstm=3,num_classes=4):
        super(ResNet50Pos,self).__init__()
        self.resnet50=models.resnet50()
        self.resnet50.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.resnet50.layer4[2].conv2=nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=2,dilation=2,bias=False)
        self.conv_lstm=ConvLSTMCell(2048*2-256,512)
        self.conv=nn.Conv2d(19,512,3,padding=1)
        self.num_conv_lstm=num_conv_lstm
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(in_features=512,out_features=num_classes,bias=True)
        self.FPA1=FeaturePyramidAttention(256)
        self.FPA2=FeaturePyramidAttention(512)
        self.FPA3=FeaturePyramidAttention(1024)
        self.FPA4=FeaturePyramidAttention(2048)
        self.upsamp4=nn.UpsamplingBilinear2d(scale_factor=8)
        self.upsamp3=nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsamp2=nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,x,embedding):
        x=self.resnet50.conv1(x)
        x=self.resnet50.bn1(x)
        x=self.resnet50.relu(x)
        x=self.resnet50.maxpool(x)
        x1=self.resnet50.layer1(x)
        x2=self.resnet50.layer2(x1)
        x3=self.resnet50.layer3(x2)
        x4=self.resnet50.layer4(x3)
        x1=self.FPA1(x1)
        x2=self.FPA2(x2)
        x2=self.upsamp2(x2)
        x3=self.FPA3(x3)
        x3=self.upsamp3(x3)
        x4=self.FPA4(x4)
        x4=self.upsamp4(x4)
        x=torch.cat((x1,x2,x3,x4),dim=1)
        embedding=embedding.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,64,64)
        embedding=self.conv(embedding)
        hidden=embedding
        cell=embedding
        for i in range(self.num_conv_lstm):
            hidden,cell=self.conv_lstm(x,[hidden,cell])
        x=hidden
        x=self.avgpool(x)
        x=x.squeeze()
        x=self.fc(x)
        return x


class SimAM(nn.Module):
    def __init__(self,lambda_=0.0001):
        super(SimAM,self).__init__()
        self.lambda_=lambda_
    def forward(self,x):
        n=x.size(2)*x.size(3)-1
        # square of (t - u)
        d=(x-x.mean(dim=[2,3]))**2
        # d.sum() / n is channel variance
        v=d.sum(dim=[2,3])/n
        # E_inv groups all importance of X
        E_inv=d/(4*(v+self.lambda_))+0.5
        # return attended features
        return x*F.sigmoid(E_inv)


class ResNet50SimAM(nn.Module):
    def __init__(self,num_classes=4):
        super(ResNet50SimAM,self).__init__()
        self.resnet50=models.resnet50()
        self.resnet50.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.resnet50.layer4[2].conv2=nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=2,dilation=2,bias=False)
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(in_features=512,out_features=num_classes,bias=True)
        self.SimAM=SimAM()


    def forward(self,x):
        x=self.resnet50.conv1(x)
        x=self.resnet50.bn1(x)
        x=self.resnet50.relu(x)
        x=self.resnet50.maxpool(x)
        x=self.resnet50.layer1(x)
        x=self.SimAM(x)
        x=self.resnet50.layer2(x)
        x=self.SimAM(x)
        x=self.resnet50.layer3(x)
        x=self.SimAM(x)
        x=self.resnet50.layer4(x)
        x=self.SimAM(x)
        x=self.avgpool(x)
        x=x.squeeze()
        x=self.fc(x)
        return x


class MultiheadedDecoder(nn.Module):
    def __init__(self,num_classes,num_layers,base_num,ratio=7/8):
        super(MultiheadedDecoder,self).__init__()
        self.conv=[]
        self.head1=[]
        self.head2=[]
        self.head3=[]
        self.head4=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**(i)),base_num*(2**(i-1))*ratio,1))
            self.head1.append(DeconvBlock(base_num*(2**(i)),base_num*(2**(i-1))*(1-ratio),1))
            self.head2.append(DeconvBlock(base_num*(2**(i)),base_num*(2**(i-1))*(1-ratio),1))
            self.head3.append(DeconvBlock(base_num*(2**(i)),base_num*(2**(i-1))*(1-ratio),1))
            self.head4.append(DeconvBlock(base_num*(2**(i)),base_num*(2**(i-1))*(1-ratio),1))
        self.conv.append(nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=1,stride=1,padding=0))
        self.conv=nn.Sequential(*self.conv)
        self.head1=nn.Sequential(*self.head1)
        self.head2=nn.Sequential(*self.head2)
        self.head3=nn.Sequential(*self.head3)
        self.head4=nn.Sequential(*self.head4)


    def forward(self,x,b,cls):
        cls=cls.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        for i in range(self.num_layers):
            if i!=self.num_layers-1:
                x1=self.head1[i](x,b[i])
                x2=self.head2[i](x,b[i])
                x3=self.head3[i](x,b[i])
                x4=self.head4[i](x,b[i])
                x=self.conv[i](x,b[i])
                x1=x1.unsqueeze(1)
                x2=x2.unsqueeze(1)
                x3=x3.unsqueeze(1)
                x4=x4.unsqueeze(1)
                x_=torch.cat((x1,x2,x3,x4),dim=1)
                x_=x_*cls
                x_=torch.sum(x_,dim=1)
                x=torch.cat((x,x_),dim=1)
            else:
                x=self.conv[i](x)
        return x

class MultiheadedUNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,base_num=64):
        super(MultiheadedUNet,self).__init__()
        self.encoder=Encoder(input_channels,num_layers,base_num)
        self.decoder=MultiheadedDecoder(num_classes,num_layers,base_num)
    def forward(self,x,cls):
        x,b=self.encoder(x)
        x=self.decoder(x,b,cls)
        return x

class DirichletResNet(nn.Module):
    def __init__(self,input_channels=1,num_classes=4,resnet_type='resnet18'):
        super(DirichletResNet,self).__init__()
        if resnet_type=='resnet50':
            self.resnet=models.resnet50()
            self.resnet.fc=nn.Linear(in_features=2048,out_features=num_classes,bias=True)
        if resnet_type=='resnet18':
            self.resnet=models.resnet18()
            self.resnet.fc=nn.Linear(in_features=512,out_features=num_classes,bias=True)
        self.resnet.conv1=nn.Conv2d(input_channels,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.num_classes=num_classes

    def forward(self,x):
        x=self.resnet(x)
        evidence=torch.exp(torch.clamp(x,-10,10))
        alpha=evidence+1
        S=torch.sum(alpha,dim=-1,keepdim=True)
        uncertainty=self.num_classes/S
        prob=alpha/S
        return prob,uncertainty,S,alpha,evidence



class AE(nn.Module):
    def __init__(self,input_channels,num_layers,mode='normal',base_num=64,dropout_rate=0):
        super(AE,self).__init__()
        self.dropout_rate=dropout_rate
        self.encoder=AEEncoder(input_channels,num_layers,base_num,dropout_rate)
        self.mode=mode
        self.decoder=AEDecoder(num_layers,base_num,dropout_rate=dropout_rate)
        if mode=='normal':
            self.last=nn.Conv2d(base_num,1,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False)
        elif mode=='uncertainty':
            self.last=nn.Conv2d(base_num,4,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        x=self.last(x)
        if self.mode=='normal':
            return x
        elif self.mode=='uncertainty':
            out=torch.zeros_like(x)
            out[:,:3,:,:]=F.softplus(x[:,:3,:,:])
            out[:,0:1,:,:]+=1
            out[:,3:4,:,:]=x[:,3:4,:,:]
            return out #x:alpha beta nu gamma




