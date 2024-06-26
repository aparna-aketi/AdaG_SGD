
"""
Distributed Gossip Wrapper

:description: Multi-Threaded Gossip Model Wrapper; designed for efficient
              multi-peer training.
"""

import functools
import time
import sys
import threading
import copy
import os

import torch
import torch.distributed as dist
from torch.cuda.comm import broadcast_coalesced, reduce_add_coalesced
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from .gossiper import SGD_DS
from .mixing_manager import UniformMixing
from .utils import (
    create_process_group, communicate, flatten_tensors,
    group_by_dtype, make_logger, unflatten_tensors, quantize_tensor, quantize_layerwise, sparsify_layerwise, unsparsify_layerwise)

from .compressor import QuantizationCompressor, SparsificationCompressor

HEARTBEAT_TIMEOUT = 1000  # maximum time to wait for message (seconds)


class GossipDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(self, module, device_ids=None, rank=None, world_size=None,
                 graph=None, mixing=None, comm_device=None, 
                 overlap=False, synch_freq=0, verbose=False, use_streams=False,
                 level=32, biased = False, gamma = 1.0, beta=0.999, qgm=False, ada_gamma=False,
                 compress_ratio=0.5, compress_fn = 'sparsify', compress_op = 'top_k'):
        super(GossipDataParallel, self).__init__()

        # devices available locally
        if device_ids is None:
            device_ids     = list(range(torch.cuda.device_count()))
        self.output_device = device_ids[0]
        self.device_ids    = device_ids
    
        if world_size is None or rank is None:
            assert dist.is_initialized()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.process_rank = rank
        

        # put model on output device
        self.module    = module
        first_param_dtype = next(self.module.parameters()).dtype
        
        # prepare local intra-node all-reduce objects
        self._module_copies = [self.module]

        # choose communication device based on backend
        if comm_device is None:
            cpu_comm = True if dist.get_backend() == 'gloo' else False
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
        self.__cpu_comm = comm_device.type == 'cpu'
        

        if mixing is None:
            mixing = UniformMixing(graph, comm_device)
        self.mixing_weights   = mixing.get_mixing_weights()
        self.averaging_rate   = torch.ones(1, device=comm_device).type(first_param_dtype)*gamma
        self.beta            = torch.ones(1, device=comm_device).type(first_param_dtype)*beta
        self.ada_gamma        = ada_gamma
        self.qgm              = qgm

        # distributed backend config
        self.dist_config = {
            'verbose':      verbose,
            'comm_device':  comm_device,
            'graph':        graph,
            'rank':         rank,
            'process_rank': self.process_rank,
            'world_size':   world_size,
            'cpu_comm':     self.__cpu_comm,
            'level':        level,
            'biased':       biased,
            'compressor':   compress_fn,
            'ratio':        compress_ratio,
            'op':           compress_op,
            'data_transferred': 0,
            't': 0}
            
        self.overlap = overlap
        self.synch_freq = synch_freq
        self.num_updates = 0
        self.asynch = synch_freq > 0

        # logger used to print to stdout
        self.logger = make_logger(rank, verbose)
        

        # prepare parameters for gossip
        self.gossip_enable = True
        self.gossiping = False
        self.params_mixed = True
        
        self.gossip_params        = []
        self.gossip_device_buffer = []
        self.neighbor_copy        = []
        self.self_xhat            = []
        self.avg_err_sq           = []
      
        for p in module.parameters():
            cp = p.clone().detach_()
            cp = cp.cpu().pin_memory() if self.__cpu_comm else cp.to(comm_device)
            self.gossip_params.append(cp)
            self.gossip_device_buffer.append(cp)
            self.neighbor_copy.append(torch.zeros_like(cp).to(comm_device))
            self.self_xhat.append(torch.zeros_like(cp).to(comm_device))
            self.avg_err_sq.append(torch.zeros_like(cp).to(comm_device))
          
        # prepare gossip process control objects
        self.gossip_lock = threading.Lock()
        self.gossip_flag = threading.Event()
        self.train_flag = threading.Event()

        if self.dist_config['comm_device'].type != 'cpu' and use_streams:
            self.gossip_stream = torch.cuda.Stream()
        else:
            self.gossip_stream = torch.cuda.current_stream(device=comm_device)

        
        self.gossip_thread = threading.Thread(
                target=GossipDataParallel._gossip_target,
                args=(self.dist_config,
                      self.gossip_flag,
                      self.train_flag,
                      self.gossip_lock,
                      self.gossip_params,
                      self.gossip_device_buffer,
                      self.gossip_stream,
                      self.self_xhat,))
        self.gossip_thread.daemon = True
        self.gossip_thread.name = 'Gossip-Thread'
        self.gossip_thread.start()
      
        # wait for thread to complete initialization
        self.gossip_flag.wait()
        self.gossip_flag.clear()

        # register ps/grad-reduction hooks
        self.__register_hooks()

    def update_gossiper(self, attr, val):
        self.logger.debug('waiting for gossip lock')
        with self.gossip_lock:
            self.logger.debug('gossip lock received')
            for gossiper in self.dist_config['gossipers'].values():
                if val == getattr(gossiper, attr):
                    self.logger.debug('nothing to update')
                    return
                # update attr
                self.logger.debug('setting gossiper {} to {}'.format(attr, val))
                setattr(gossiper, attr, val)

    def state_dict(self, finish_gossip=True):
        if finish_gossip:
            self._query_gossip_queue()

        super_dict = super(GossipDataParallel, self).state_dict()
        supplanted_dict = {'state_dict': super_dict,  }
        return supplanted_dict

    def load_state_dict(self, load_dict):
        state_dict = load_dict['state_dict']
        super(GossipDataParallel, self).load_state_dict(state_dict)
            
            
    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module(*inputs[0], **kwargs[0])
        
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=0)


    def train(self, mode=True):
        super(GossipDataParallel, self).train(mode)
        self.gossip_enable = True
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(GossipDataParallel, self).eval()
        self.gossip_enable = False
        for module in self._module_copies[1:]:
            module.eval()
        self._query_gossip_queue(non_blocking=self.asynch)

    def block(self):
        self.logger.info('blocking')
        dist.barrier()
    
    def update_hats(self):
        for xj_hat, xi_hat, qj in zip(self.neighbor_copy, self.self_xhat, self.gossip_device_buffer): # change to have weighted q
            xj_hat.data.add_(qj.data)
            qj.data.copy_(xj_hat.data)
            qj.data.add_(xi_hat.data, alpha = float(self.mixing_weights['uniform'])-1.0)

    def _query_gossip_queue(self, non_blocking=False):
        """ Check gossip-queue for push-sum residuals and update model """
        if not self.gossip_enable:
            return

        self.logger.debug('querying gossip queue')

        # no gossip happening right now so just return
        if not self.gossiping:
            #self.logger.warning('not gossiping right now')
            return False

        if not non_blocking:
            if not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT):
                raise NameError('Gossip flag timeout')
                sys.exit()  # HEARTBEAT monitor

        # query gossip thread
        if self.gossip_flag.is_set():
            self.logger.debug('received gossip flag')
            
            self.update_hats()
              
            for p, r, v in zip(self.module.parameters(), self.gossip_device_buffer, self.avg_err_sq):
                if self.ada_gamma:
                    # compute the adaptive scaling
                    v.mul_(self.beta).addcmul_(r.data, r.data, value = 1.0 - float(self.beta))
                    buf = ((copy.deepcopy(v).div_(1.0 - (float(self.beta)**self.dist_config['t']))).sqrt()).add_(1e-8)
                    exp = torch.ones_like(buf)*self.averaging_rate.data
                    exp.data.div_(buf.data)
                    exp = torch.clamp(exp, max=1.0)
                    #apply the scaling to r.data
                    r.data.mul_(exp.type(r.data.dtype))
                else:
                    r.data.mul_(self.averaging_rate.type(r.data.dtype))
                p.data.add_(r.data)  

                #qgm related stuff
                if self.qgm:
                    self.optimizer.state[p]['momentum_buffer'].data.add_(r.data, alpha=-1.0/self.lr)
     
            # update flags
            self.logger.debug('updated model params')
            self.gossip_flag.clear()
            self.params_mixed = True
            self.gossiping = False
            return True

    def transfer_params(self, optimizer, lr=0.1):
        """ Transfers COPY of model parameters to gossip queue """
        self.lr        = lr
        self.optimizer = optimizer
        if (not self.gossip_enable ):
            return False, 0

        self.logger.debug('transfering model params')

        # don't transfer new params if old params haven't been mixed yet
        if not self.params_mixed:
            self.logger.warning('params not mixed')
            return False, 0
        # params gpu-gpu copy (fast)
        for xi, xi_hat, gossip_device_buffer_elem in zip(self.module.parameters(), self.self_xhat, self.gossip_device_buffer):
            gossip_device_buffer_elem.data.copy_(xi.data)
            gossip_device_buffer_elem.data.add_(xi_hat.data, alpha=-1)
             
        # --
        # buffer to gossip-thread copy (potentially slow, but asynchronous)
        # --
        self.gossip_stream.wait_stream(torch.cuda.current_stream(device = self.dist_config['comm_device']))
        
        with torch.cuda.stream(self.gossip_stream):
            for b, gp in zip(self.gossip_device_buffer, self.gossip_params):
                gp.copy_(b, non_blocking=True)
        
        self.dist_config['t']+=1
        # update flags
        self.logger.debug('transfered model params')
        self.params_mixed = False
        self.gossiping = True
        self.train_flag.set()
        return True, self.dist_config['data_transferred']

    @staticmethod
    def _gossip_into_receive_buffer(send_buffer, 
                                    gossiper, 
                                    receive_buffer,
                                    gossip_lock,
                                    dist_config, self_xhat):
     
        if dist_config['compressor'] == 'quantize':
            comp_unflat   = quantize_layerwise(send_buffer, QuantizationCompressor(), quantization_level= dist_config['level'], is_biased = dist_config['biased']) # C(Vt)
            comp_msg      = flatten_tensors(comp_unflat)
            shapes        = None
            uncompress    = False
            
        elif dist_config['compressor'] == 'sparsify':
            comp_msg, shapes = sparsify_layerwise(send_buffer, SparsificationCompressor(), dist_config['op'], dist_config['ratio'], dist_config['biased'])           
            comp_unflat      = unsparsify_layerwise(comp_msg, shapes, send_buffer)
            uncompress   = True
        else:
            raise NotImplementedError
             
        # update xi_hat
        for xi_hat, qi in zip(self_xhat, comp_unflat):
            xi_hat.data.add_(qi.data)
          
        with gossip_lock:
            data_amt = 0
            #in_msg = sum of q_j's
            in_msg, data_amt = gossiper.mix(comp_msg, send_buffer, uncompress=uncompress, shapes=shapes)
            dist_config['data_transferred'] = data_amt
         
        for r, g in zip(unflatten_tensors(in_msg, send_buffer), receive_buffer):
            if dist_config['cpu_comm']:
                g.copy_(r, non_blocking=True)
            else:
                g.data.copy_(r)           
        return 

    @staticmethod
    def _gossip_target(dist_config, gossip_flag, train_flag, gossip_lock,
                       gossip_params, gossip_device_buffer,
                       gossip_stream, self_xhat):
        """ Gossip thread, which performs push-sum on model params """
        logger = make_logger(dist_config['rank'], dist_config['verbose'])

        gossip_params_by_dtype        = group_by_dtype(gossip_params) #send
        gossip_device_buffer_by_dtype = group_by_dtype(gossip_device_buffer) #receive
        self_xhat_by_dtype            = group_by_dtype(self_xhat) #xi_hat

        gossipers = {}
        # init gossip instance
        gossiper_class = SGD_DS
        for dtype in gossip_params_by_dtype:
            gossipers[dtype] = gossiper_class(
                flatten_tensors(gossip_params_by_dtype[dtype]), 
                device=dist_config['comm_device'],
                graph=dist_config['graph'],
                rank=dist_config['process_rank'],
                world_size=dist_config['world_size'], 
                logger=logger)

        dist_config['gossipers'] = gossipers
        gossip_flag.set()

        # gossip loop
        while True:
            train_flag.wait()
            logger.debug('received train-flag')
            try:
                if True:
                    for dtype in gossip_params_by_dtype:
                        GossipDataParallel._gossip_into_receive_buffer(
                            gossip_params_by_dtype[dtype], gossipers[dtype],
                            gossip_device_buffer_by_dtype[dtype], 
                            gossip_lock, dist_config, self_xhat_by_dtype[dtype])
                    
            except RuntimeError as e:
                logger.warning('received runtime error {}'.format(e))
                for gossiper in gossipers.values():
                    gossiper.clean_msg_buffers_()
                
            finally:
                # Make sure all queued operations are complete
                gossip_stream.synchronize()
                # give main thread go-ahead to read our gossip buffer
                train_flag.clear()
                gossip_flag.set()

    def __register_hooks(self):
        """
        Registers push-sum de-bias/bias hooks in pre-forward/post-backward
        passes in all leaf modules
        """
        self.register_forward_pre_hook(self.__make_forward_pre_hook())


    def __make_forward_pre_hook(self):
        self.logger.debug('making forward pre-hook')

        def hook(*unused):
            """ Query gossip queue and de-bias during forward pass """
            # gossip during training (not inference)
            if self.gossip_enable:
                non_blocking = self.num_updates < self.synch_freq
                if self._query_gossip_queue(non_blocking):
                    self.num_updates = 0
                else:
                    self.num_updates += 1
        return hook
