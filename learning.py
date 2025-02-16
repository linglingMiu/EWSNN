from typing import Callable, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import monitor, base
import neuron


def stdp_conv2d_single_step(
    conv: nn.Conv2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
    trace_pre: Union[torch.Tensor, None], trace_post: Union[torch.Tensor, None],
    tau_pre: float, tau_post: float,
    f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if conv.dilation != (1, 1):
        raise NotImplementedError(
            'STDP with dilation != 1 for Conv2d has not been implemented!'
        )
    if conv.groups != 1:
        raise NotImplementedError(
            'STDP with groups != 1 for Conv2d has not been implemented!'
        )

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice,
                mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

    if trace_pre is None:
        trace_pre = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros_like(
            out_spike, device = in_spike.device, dtype=in_spike.dtype
        )

     
    trace_pre = trace_pre - trace_pre * math.exp(-1 / tau_pre) + in_spike * 1.5
    trace_post = trace_post - trace_post * math.exp(-1 / tau_post) + out_spike * 1.5
    #print(trace_post.shape)
    #print(in_spike.shape)
    
    #out_spike = competitive_inhibition(out_spike, k)

    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w

            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]  # shape = [batch_size, C_in, h_out, w_out]
            post_spike = out_spike  # shape = [batch_size, C_out, h_out, h_out]
            weight = conv.weight.data[:, :, h, w]   # shape = [batch_size_out, C_in]

            tr_pre = trace_pre[:, :, h:h_end:stride_h, w:w_end:stride_w]    # shape = [batch_size, C_in, h_out, w_out]
            tr_post = trace_post    # shape = [batch_size, C_out, h_out, w_out]
            #x_upsampled = F.interpolate(x, size=(34, 34), mode='bilinear', align_corners=False)
            #print(pre_spike.unsqueeze(1).shape)
            #print(tr_post.unsqueeze(2).shape)
            delta_w_pre = - (f_pre(weight) *
			                (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                            .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4]))
            #print((tr_post.unsqueeze(2) * pre_spike.unsqueeze(1)).shape)
            delta_w_post = f_post(weight) * \
                           (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2))\
                           .permute([1, 2, 0, 3, 4]).sum(dim = [2, 3, 4])
            delta_w[:, :, h, w] += delta_w_pre * 0.1 + delta_w_post * 0.01
            delta_w[:, :, h, w] -= (delta_w_post + delta_w_pre * 0.1) * 0.02
            #delta_w[:, :, h, w] += (prev_delta_w_pre * 0.1  + prev_delta_w_post * 0.01) * 0.5
            
    return trace_pre, trace_post, delta_w

class STDPLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear], sn: neuron.BaseNode,
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super().__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)
        

    def reset(self):
        super(STDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def step(self, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Conv2d):
                stdp_f = stdp_conv2d_single_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)     # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)   # [batch_size, N_out]
            
            
            self.trace_pre, self.trace_post, dw = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post, 
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w

