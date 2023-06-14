# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils_v4 import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, Bag, Light_Bag, FSM
import logging
import sys

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False



class PIDNet_v4(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet_v4, self).__init__()
        self.augment = augment
        
        # I Branch
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        
        # P Branch


        self.FSM3 = FSM(3, planes * 2, planes * 4, planes)
        self.FSM4 = FSM(4, planes * 2, planes * 8, planes)
        

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes * 2, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
            
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)           

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        
        return layer

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        
        #Stage 1,2
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        #Stage 3
        

        x_ = self.layer3_(x) # detail-stage3  
        x_d = self.layer3_d(x) # boundary-stage3
        x = self.relu(self.layer3(x)) # context-stage3         
        x_d = x_d + F.interpolate(
                        self.diff3(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        
        c_ff3, d_ff3 = self.FSM3(x, x_)
        x = x + c_ff3 # context = context + detail' 
        x_ = x_ + d_ff3 # detail = detail + context' 

        if self.augment:
            temp_p = x_
        
        #Stage 4
        x = self.relu(self.layer4(x)) # context-stage4
        x_ = self.layer4_(self.relu(x_)) # detail-stage4 
        x_d = self.layer4_d(self.relu(x_d)) #boundary- stage4
        x_d = x_d + F.interpolate(
                        self.diff4(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)


        c_ff4, d_ff4 = self.FSM4(x, x_)
        x = x + c_ff4 # context = context + detail'
        x_ = x_ + d_ff4 # detail = detail + context'
        temp_d = x_d
        
        #Stage 5
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
                        self.spp(self.layer5(x)),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment: 
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_      

def get_seg_model(cfg, imgnet_pretrained):
    
    if 's' in cfg.MODEL.NAME:
        model = PIDNet_v4(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNet_v4(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet_v4(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True)
    
    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict'] 
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)
    
    return model

def get_pred_model(name, num_classes):
    
    if 's' in name:
        model = PIDNet_v4(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet_v4(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet_v4(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)
    
    return model

if __name__ == '__main__':
    
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    sys.path.append('/home/mvpserverfive/minseok/PIDNet/models/')
    device = torch.device('cuda')
    model = get_pred_model(name='pidnet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    
    
    


class Local2GlobalAttn(nn.Module):
    def __init__(
        self,
        inp,
        token_dim=128,
        token_num=6,
        inp_res=0,
        norm_pos='post',
        drop_path_rate=0.
    ):
        super(Local2GlobalAttn, self).__init__()

        num_heads = 2
        self.scale = (inp // num_heads) ** -0.5

        self.q = nn.Linear(token_dim, inp)
        self.proj = nn.Linear(inp, token_dim)
        
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)


    def forward(self, x):
        features, tokens = x
        bs, C, _, _ = features.shape

        t = self.q(tokens).permute(1, 0, 2) # from T x bs x Ct to bs x T x Ct
        k = features.view(bs, C, -1)        # bs x C x HW
        attn = (t @ k) * self.scale

        attn_out = attn.softmax(dim=-1)             # bs x T x HW
        attn_out = (attn_out @ k.permute(0, 2, 1))  # bs x T x C
                                                    # note here: k=v without transform
        t = self.proj(attn_out.permute(1, 0, 2))    #T x bs x C

        tokens = tokens + self.drop_path(t)
        tokens = self.layer_norm(tokens)

        return tokens
    
class Global2Local(nn.Module):
    def __init__(
        self,
        inp,
        inp_res=0,
        block_type='mlp',
        token_dim=128,
        token_num=6,
        attn_num_heads=2,
        use_dynamic=False,
        drop_path_rate=0.,
        remove_proj_local=True, 
    ):
        super(Global2Local, self).__init__()
        print(f'G2L: {attn_num_heads} heads, inp: {inp}, token: {token_dim}')

        self.token_num = token_num
        self.num_heads = attn_num_heads
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, inp),
                h_sigmoid(),
            )


        if 'mlp' in self.block:
            self.mlp = nn.Linear(token_num, inp_res)

        if 'attn' in self.block:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.k = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(token_dim, inp)
        self.drop_path = DropPath(drop_path_rate)

        self.remove_proj_local = remove_proj_local
        if self.remove_proj_local == False:
            self.q = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.fuse = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
 
    def forward(self, x):
        out, tokens = x

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            v = self.proj(tokens)
            v = (v * alp).permute(1, 2, 0)
        else:
            v = self.proj(tokens).permute(1, 2, 0)  # from T x bs x Ct -> T x bs x C -> bs x C x T 

        bs, C, H, W = out.shape
        if 'mlp' in self.block:
            g_sum = self.mlp(v).view(bs, C, H, W)       # bs x C x T -> bs x C x H x W

        if 'attn' in self.block:
            if self.remove_proj_local:
                q = out.view(bs, self.num_heads, -1, H*W).transpose(-1, -2)                         # bs x N x HW x C/N
            else:
                q = self.q(out).view(bs, self.num_heads, -1, H*W).transpose(-1, -2)                         # bs x N x HW x C/N

            k = self.k(tokens).permute(1, 2, 0).view(bs, self.num_heads, -1, self.token_num)    # from T x bs x Ct -> bs x C x T -> bs x N x C/N x T
            attn = (q @ k) * self.scale                         # bs x N x HW x T

            attn_out = attn.softmax(dim=-1)                     # bs x N x HW x T
            
            vh = v.view(bs, self.num_heads, -1, self.token_num) # bs x N x C/N x T
            attn_out = (attn_out @ vh.transpose(-1, -2))        # bs x N x HW x C/N
                                                                # note here k != v
            g_a = attn_out.transpose(-1, -2).reshape(bs, C, H, W)   # bs x C x HW

            if self.remove_proj_local == False:
                g_a = self.fuse(g_a)            

            g_sum = g_sum + g_a if 'mlp' in self.block else g_a

        out = out + self.drop_path(g_sum)

        return out
