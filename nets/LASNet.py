import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter
from .darknet import BaseConv
from .RDIAN.segmentation import RDIAN


class LASNet(nn.Module):
    def __init__(self, num_classes, num_frame=5):
        super(LASNet, self).__init__()
        
        self.num_frame = num_frame
        self.backbone = RDIAN()
        self.MAF = Motion_Affinity_Fusion_Module(channels=[128], num_frame=num_frame)
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")
        self.mapping0 = nn.Sequential(
                nn.Conv2d(128*num_frame, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU())
        self.LAS = Linking_Aware_Sliced_Module(input_dim=64, hidden_dim=[64,64], kernel_size=(3, 3), num_layers=2, num_slices=2, num_frames=self.num_frame)
        self.mapping1 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU()) 
        self.mapping2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU())
        self.conv_backbone = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU()) 
        self.motion_head = nn.Sequential(
            BaseConv(self.num_frame*128,128,3,1),
            BaseConv(128,64,3,1),
            BaseConv(64,1,1,1)) 
        self.mm_loss = motion_mask_loss()
       
        
    def forward(self, inputs, multi_targets=None): 
        feat = []
        for i in range(self.num_frame):
            feats = self.backbone(inputs[:,:,i,:,:]) 
            feats = self.conv_backbone(feats) 
            feat.append(feats)
        
        multi_feat =  torch.stack([self.mapping1(feat[i]) for i in range(self.num_frame)], 1) 
        lstm_output, _ = self.LAS(multi_feat)
        motion_relation = lstm_output[-1] 
        motion = torch.stack([self.mapping2(motion_relation[:,i,:,:,:]) for i in range(self.num_frame)], 1) 
        feat = self.MAF(feat, motion) 
        outputs  = self.head(feat) 
        
        if self.training:
            pred_m = self.motion_head(torch.cat([motion[:,i,:,:,:] for i in range(self.num_frame)], 1)) 
            pred_m = F.interpolate(pred_m, size=[inputs.shape[2], inputs.shape[3]], mode='bilinear', align_corners=True)
            mm_loss = self.mm_loss(pred_m, multi_targets)

        if self.training:
            return  outputs, mm_loss
        else:
            return  outputs

class motion_mask_loss(nn.Module):
    def __init__(self):
        super(motion_mask_loss, self).__init__()
        
    def forward(self, pred_m, multi_targets):
        multi_targets = np.array(multi_targets)
        gt_target = torch.tensor(multi_targets)
        heatmap = torch.zeros(pred_m.shape[0], pred_m.shape[2], pred_m.shape[3]).cuda() 
        for b in range(gt_target.shape[0]):
            for f in range(gt_target.shape[1]):
                for t in range(gt_target.shape[2]):
                    x, y = gt_target[b,f,t,:2]
                    s_x, s_y = gt_target[b,f,t,2:4]
                    heatmap[b,int(x):int(x) + int(s_x), int(y):int(y) + int(s_y) ] = 255
        target = heatmap.unsqueeze(1)
        pred = torch.sigmoid(pred_m)
        smooth = 1
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        loss = 1 - torch.mean(loss)
        return loss


class Linking_Aware_Node(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(Linking_Aware_Node, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) 
        self.conv2 = nn.Conv2d(in_channels=4 * self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) 

    def forward(self, input_tensor, input_head, cur_state, multi_head): 
        h_cur, c_cur = cur_state 
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined) 
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i) 
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g       
        h_next = o * torch.tanh(c_next) 
        
        m_h, m_c = multi_head
        combined2 = torch.cat([input_tensor, h_cur, input_head, m_h], dim=1) 
        combined_conv2 = self.conv2(combined2) 
        mm_i, mm_f, mm_o, mm_g = torch.split(combined_conv2, self.hidden_dim, dim=1)
        
        m_i = torch.sigmoid(mm_i+cc_i) 
        m_f = torch.sigmoid(mm_f+cc_f)
        m_o = torch.sigmoid(mm_o+cc_o)
        m_g = torch.tanh(mm_g+cc_g)
        
        m_c_next = m_f * m_c + m_i * m_g      
        m_h_next = m_o * torch.tanh(m_c_next)
 
        return h_next, c_next, m_h_next, m_c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
class Linking_Aware_Sliced_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, num_slices, num_frames,
                 batch_first=True, bias=True, return_all_layers=False):
        super(Linking_Aware_Sliced_Module, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.deep = num_slices
        self.frames = num_frames
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        Node_list = {}
       
        for i in range(self.deep):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            for j in range(self.num_layers):
                Node_list.update({'%d%d'%(i,j): Linking_Aware_Node(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[j],
                                          kernel_size=self.kernel_size[j],
                                          bias=self.bias)}) 
        self.Node_list = nn.ModuleDict(Node_list)
        
        self.linking_weight1 = {}
        for i in range(0,self.deep):
            for j in range(0,self.num_layers):
                for k in range(0,self.frames): 
                    self.linking_weight1.update({'%d%d%d'%(i,j,k): nn.Conv3d(self.frames,self.frames,1,1,0)})
        self.linking_1 = nn.ModuleDict(self.linking_weight1)
                
        self.linking_weight2 = {}
        for i in range(0,self.deep):
            for j in range(0,self.num_layers):
                for k in range(0,self.frames):
                    self.linking_weight2.update({'%d%d%d'%(i,j,k):  nn.Conv3d(self.frames,self.frames,1,1,0)})
        self.linking_2 = nn.ModuleDict(self.linking_weight2)

        self.state_weight1 = {}             
        for i in range(0,self.deep):
            for j in range(0,self.num_layers):
                for t in range(1,self.frames):
                    self.state_weight1.update({'%d%d%d%d'%(i,j,t,1): nn.Conv3d(t+1,t+1,1,1,0)})
                    self.state_weight1.update({'%d%d%d%d'%(i,j,t,2): nn.Conv3d(t+1,t+1,1,1,0)})
        self.state_1 = nn.ModuleDict(self.state_weight1) 

        self.state_weight2 = {} 
        for i in range(1,self.deep):
            for j in range(0,self.num_layers):
                for t in range(0,self.frames):
                    self.state_weight2.update({'%d%d%d%d'%(i,j,t,1): nn.Conv3d(i+1,i+1,1,1,0)})
                    self.state_weight2.update({'%d%d%d%d'%(i,j,t,2): nn.Conv3d(i+1,i+1,1,1,0)})
        
        self.state_2 = nn.ModuleDict(self.state_weight2) 
        

    def forward(self, input_tensor, hidden_state=None):
        
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()
        
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
            deep_state = self._init_motion_hidden(batch_size=b,
                                             image_size=(h, w), t_len = input_tensor.shape[1])

        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        head_input = input_tensor

        input_deep_h = {}
        input_deep_c = {}
        slice_state = {}
        
        for deep_idx in range(self.deep):  

            for layer_idx in range(self.num_layers):
                
                past_state = []
                output_inner = []
                h, c  = hidden_state['%d%d'%(deep_idx,layer_idx)] 
                
                for t in range(seq_len): 

                    cur_input = self.linking_1['%d%d%d'%(deep_idx,layer_idx,t)](cur_layer_input)
                    cur_input2 = nn.functional.softmax(cur_input, dim=1)
                    selected_cur_input_index = torch.argmax(cur_input2[0,:,0,0,0])
                    cur_input = cur_input[:,selected_cur_input_index,:,:,:]
                   
                    s_input = self.linking_2['%d%d%d'%(deep_idx,layer_idx,t)](head_input)
                    s_input2 = nn.functional.softmax(s_input, dim=1)
                    selected_s_input_index = torch.argmax(s_input2[0,:,0,0,0])
                    s_input = s_input[:,selected_s_input_index,:,:,:]
                    
                    if t ==0:
                        past_state.append([h,c])
                        selected_state = [h,c]
                    else:
                        p_h=[]
                        p_c=[]
                        for i in range(len(past_state)):
                            past_h, past_c = past_state[i]
                            p_h.append(past_h)
                            p_c.append(past_c)
                        pp_h = torch.stack([p_h[i] for i in range(len(p_h))], 1)
                        pp_c = torch.stack([p_c[i] for i in range(len(p_c))], 1)

                        pp_h = self.state_1['%d%d%d%d'%(deep_idx,layer_idx,t,1)](pp_h)
                        pp_h2 = nn.functional.softmax(pp_h, dim=1)
                        selected_pp_h_index = torch.argmax(pp_h2[0,:,0,0,0])        
                        pp_h = pp_h[:,selected_pp_h_index,:,:,:]
                        
                        pp_c = self.state_1['%d%d%d%d'%(deep_idx,layer_idx,t,2)](pp_c)
                        pp_c2 = nn.functional.softmax(pp_c, dim=1)
                        selected_pp_c_index = torch.argmax(pp_c2[0,:,0,0,0])
                        pp_c = pp_c[:,selected_pp_c_index,:,:,:]

                        selected_state = [pp_h,pp_c]

                    if deep_idx == 0:
                        m_h, m_c = deep_state['%d%d'%(layer_idx, t)] 
                    else:
                        m_h = input_deep_h['%d%d%d'%(deep_idx-1,layer_idx, t)]
                        m_c = input_deep_c['%d%d%d'%(deep_idx-1,layer_idx, t)]

                    slice_state.update({'%d%d%d'%(deep_idx,layer_idx,t): [m_h, m_c]})
                    
                    if deep_idx == 0:
                        selected_slice = slice_state['%d%d%d'%(deep_idx,layer_idx,t)]
                    else:
                        mm_h=[]
                        mm_c=[]
                        for i in range(deep_idx+1):
                            past_mh, past_mc = slice_state['%d%d%d'%(i,layer_idx,t)]
                            mm_h.append(past_mh)
                            mm_c.append(past_mc)
                        pp_mh = torch.stack([mm_h[i] for i in range(len(mm_h))], 1)
                        pp_mc = torch.stack([mm_c[i] for i in range(len(mm_c))], 1)

                        pp_mh = self.state_2['%d%d%d%d'%(deep_idx,layer_idx,t,1)](pp_mh)
                        pp_mc = self.state_2['%d%d%d%d'%(deep_idx,layer_idx,t,2)](pp_mc)

                        pp_mh2 = nn.functional.softmax(pp_mh, dim=1)
                        selected_pp_mh_index = torch.argmax(pp_mh2[0,:,0,0,0])
                        pp_mh = pp_mh[:,selected_pp_mh_index,:,:,:]

                        pp_mc2 = nn.functional.softmax(pp_mc, dim=1)
                        selected_pp_mc_index = torch.argmax(pp_mc2[0,:,0,0,0])
                        pp_mc = pp_mc[:,selected_pp_mc_index,:,:,:]
               
                        selected_slice = [pp_mh,pp_mc]
                    
                    h, c, m_h, m_c = self.Node_list['%d%d'%(deep_idx,layer_idx)](input_tensor=cur_input, input_head = s_input,  cur_state=selected_state, multi_head=selected_slice) 
                    
                    past_state.append([h,c])
                    output_inner.append(h+m_h)
                    
                    input_deep_h.update({'%d%d%d'%(deep_idx,layer_idx,t): m_h}) 
                    input_deep_c.update({'%d%d%d'%(deep_idx,layer_idx,t): m_c}) 

                layer_output = torch.stack(output_inner, dim=1) 
                head_output = torch.stack(([input_deep_h['%d%d%d'%(deep_idx, layer_idx, t)] for t in range (seq_len)]), dim=1)

                cur_layer_input = layer_output
                head_input = head_output 
            
                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:]
                last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    
    def _init_hidden(self, batch_size, image_size):
        init_states = {}
        for i in range(0,self.deep):
            for j in range(0,self.num_layers):
                init_states.update({'%d%d'%(i,j): self.Node_list['%d%d'%(i,j)].init_hidden(batch_size, image_size)}) 
        return init_states
        
    def _init_motion_hidden(self, batch_size, image_size, t_len):
        
        init_states = {}
        for i in range(0,self.num_layers):
            for j in range(0,t_len):
                init_states.update({'%d%d'%(i,j): self.Node_list['00'].init_hidden(batch_size, image_size)}) 
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Motion_Affinity_Fusion_Module(nn.Module):
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        self.weight = nn.ParameterList(torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True) for _ in range(num_frame))
        
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        
        self.conv_gl = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
       
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

    def forward(self, feats, motion):
        f_feats = []
        r_feat = torch.cat([feats[i] for i in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))

        r_feats = torch.stack([self.conv_gl(torch.cat([motion[:,i,:,:,:], feats[-1]], dim=1))*self.weight[i] for i in range(self.num_frame)], dim=1)
        r_feat= self.conv_gl_mix(torch.sum(r_feats, dim=1))
        c_feat = self.conv_final(torch.cat([r_feat,c_feat], dim=1))
        f_feats.append(c_feat)

        return f_feats

    

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        
        outputs = []
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)
            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)
            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
