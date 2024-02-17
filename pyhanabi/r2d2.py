# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Tuple, Dict
from net import FFWDNet, PublicLSTMNet, LSTMNet
import time


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = [
        "vdn",
        "multi_step",
        "gamma",
        "eta",
        "boltzmann",
        "uniform_priority",
        "net",
    ]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        net,
        num_lstm_layer,
        boltzmann_act,
        uniform_priority,
        off_belief,
        greedy=False,
        nhead=None,
        nlayer=None,
        max_len=None,
        adv_type=0,
        adv_ratio=0,
        play_styles=0,
        ps_duplicate=1,
    ):
        super().__init__()
        self.play_styles = play_styles
        self.indim = in_dim
        self.ps_duplicate = ps_duplicate
        #print('Init r2d2agent, indim:', in_dim, 'play_style_embedding_dim:',self.play_style_embedding_dim)
        in_dim_list = [in_dim[0],in_dim[1]+ps_duplicate*play_styles,in_dim[2]]

        in_dim = in_dim_list

        if self.play_styles > 0:
            # generate tensors for backup 
            self.playstyle_list = torch.eye(play_styles).to(device)

        if net == "ffwd":
            self.online_net = FFWDNet(in_dim, hid_dim, out_dim).to(device)
            self.target_net = FFWDNet(in_dim, hid_dim, out_dim).to(device)
        elif net == "publ-lstm":
            self.online_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
            self.target_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        elif net == "lstm":
            self.online_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
            self.target_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        elif net == "transformer":
            self.online_net = TransformerNet(
                device, in_dim, hid_dim, out_dim, nhead, nlayer, max_len
            )
            self.target_net = TransformerNet(
                device, in_dim, hid_dim, out_dim, nhead, nlayer, max_len
            )
        else:
            assert False, f"{net} not implemented"

        for p in self.target_net.parameters():
            p.requires_grad = False

        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.net = net
        self.num_lstm_layer = num_lstm_layer
        self.boltzmann = boltzmann_act
        self.uniform_priority = uniform_priority
        self.off_belief = off_belief
        self.greedy = greedy
        self.nhead = nhead
        self.nlayer = nlayer
        self.max_len = max_len
        self.adv_type = adv_type
        self.adv_ratio = adv_ratio

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def generate_ps(self, in_tensor):
        if self.ps_duplicate == 1:
            return in_tensor
        else:
            return in_tensor.repeat_interleave(self.play_styles, dim=-1) 

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.indim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.net,
            self.num_lstm_layer,
            overwrite.get("boltzmann_act", self.boltzmann),
            self.uniform_priority,
            self.off_belief,
            self.greedy,
            adv_type=overwrite.get("adv_type", self.adv_type),
            adv_ratio=overwrite.get("adv_ratio", self.adv_ratio),
            nhead=self.nhead,
            nlayer=self.nlayer,
            max_len=self.max_len,
            play_styles=self.play_styles,
            ps_duplicate=self.ps_duplicate,
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid, legal_adv.detach()

    @torch.jit.script_method
    def boltzmann_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        temperature: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        temperature = temperature.unsqueeze(1)
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        assert adv.dim() == temperature.dim()
        logit = adv / temperature
        legal_logit = logit - (1 - legal_move) * 1e30
        assert legal_logit.dim() == 2
        prob = nn.functional.softmax(legal_logit, 1)
        action = prob.multinomial(1).squeeze(1).detach()
        return action, new_hid, prob

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        priv_s = obs["priv_s"]
        if self.play_styles > 0:
            onehot_playstyle = nn.functional.one_hot(obs["playStyle"],num_classes=self.play_styles).float()
            priv_s = torch.cat((priv_s,self.generate_ps(onehot_playstyle)),dim=-1)
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        if "eps" in obs:
            eps = obs["eps"].flatten(0, 1)
        else:
            eps = torch.zeros((priv_s.size(0),), device=priv_s.device)

        if self.vdn:
            bsize, num_player = obs["priv_s"].size()[:2]
            priv_s = priv_s.flatten(0, 1)
            publ_s = obs["publ_s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
        else:
            bsize, num_player = obs["priv_s"].size()[0], 1

        hid = {"h0": obs["h0"], "c0": obs["c0"]}

        if self.boltzmann:
            temp = obs["temperature"].flatten(0, 1)
            greedy_action, new_hid, prob = self.boltzmann_act(
                priv_s, publ_s, legal_move, temp, hid
            )
            reply = {"prob": prob}
            if self.greedy:
                action = greedy_action
            else:
                random_action = legal_move.multinomial(1).squeeze(1)
                rand = torch.rand(greedy_action.size(), device=greedy_action.device)
                assert rand.size() == eps.size()
                rand = (rand < eps).float()
                action = (greedy_action * (1 - rand) + random_action * rand).detach().long()                
        else:
            greedy_action, new_hid, legal_adv = self.greedy_act(priv_s, publ_s, legal_move, hid)
            reply = {}
            if self.greedy:
                action = greedy_action
            else:
                if self.adv_type == 0:
                    random_action = legal_move.multinomial(1).squeeze(1)
                    rand = torch.rand(greedy_action.size(), device=greedy_action.device)
                    assert rand.size() == eps.size()
                    rand = (rand < eps).float()
                    action = (greedy_action * (1 - rand) + random_action * rand).detach().long()
                
                elif self.adv_type == 1: #use sub optimal action
                    adv_mask = (legal_adv != legal_adv.max(1,keepdim=True)[0])
                    subopt_adv = adv_mask * legal_adv + 1e-3
                    subopt_adv = subopt_adv * legal_move
                    subopt_action = subopt_adv.argmax(1).detach()
                    rand = torch.rand(greedy_action.size(), device=greedy_action.device)
                    rand = (rand < self.adv_ratio).float()
                    action = (greedy_action * (1 - rand) + subopt_action * rand).detach().long()
                elif self.adv_type == 2: #use worst action
                    forbidden_act = (legal_move == 0).float()
                    worst_adv = legal_adv + 99.9*forbidden_act
                    worst_action = worst_adv.argmin(1).detach()
                    rand = torch.rand(greedy_action.size(), device=greedy_action.device)
                    rand = (rand < self.adv_ratio).float()
                    action = (greedy_action * (1 - rand) + worst_action * rand).detach().long()
                else: #use random action
                    random_action = legal_move.multinomial(1).squeeze(1)
                    rand = torch.rand(greedy_action.size(), device=greedy_action.device)
                    rand = (rand < self.adv_ratio).float()
                    action = (greedy_action * (1 - rand) + random_action * rand).detach().long()                    

        if self.vdn:
            action = action.view(bsize, num_player)
            greedy_action = greedy_action.view(bsize, num_player)
            # rand = rand.view(bsize, num_player)

        reply["a"] = action.detach().cpu()
        reply["h0"] = new_hid["h0"].detach().cpu()
        reply["c0"] = new_hid["c0"].detach().cpu()
        return reply

    @torch.jit.script_method
    def compute_target(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert self.multi_step == 1
        priv_s = input_["priv_s"]
        if self.play_styles > 0:
            onehot_playstyle = nn.functional.one_hot(input_["playStyle"],num_classes=self.play_styles).float()
            priv_s = torch.cat((priv_s,self.generate_ps(onehot_playstyle)),dim=-1)
        publ_s = input_["publ_s"]
        legal_move = input_["legal_move"]
        act_hid = {
            "h0": input_["h0"],
            "c0": input_["c0"],
        }
        fwd_hid = {
            "h0": input_["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": input_["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }
        reward = input_["reward"]
        terminal = input_["terminal"]

        if self.boltzmann:
            temp = input_["temperature"].flatten(0, 1)
            next_a, _, next_pa = self.boltzmann_act(
                priv_s, publ_s, legal_move, temp, act_hid
            )
            next_q = self.target_net(priv_s, publ_s, legal_move, next_a, fwd_hid)[2]
            qa = (next_q * next_pa).sum(1)
        else:
            next_a = self.greedy_act(priv_s, publ_s, legal_move, act_hid)[0]
            qa = self.target_net(priv_s, publ_s, legal_move, next_a, fwd_hid)[0]

        assert reward.size() == qa.size()
        target = reward + (1 - terminal) * self.gamma * qa
        return {"target": target.detach()}

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"].sum(1))}

        # swap batch_dim and seq_dim
        for k, v in input_.items():
            if k != "seq_len":
                input_[k] = v.transpose(0, 1).contiguous()

        obs = {
            "priv_s": input_["priv_s"],
            "publ_s": input_["publ_s"],
            "legal_move": input_["legal_move"],
        }
        if "playStyle" in input_.keys():
            obs["playStyle"] = input_["playStyle"]
        if self.boltzmann:
            obs["temperature"] = input_["temperature"]

        if self.off_belief:
            obs["target"] = input_["target"]

        hid = {"h0": input_["h0"], "c0": input_["c0"]}
        action = {"a": input_["a"]}
        reward = input_["reward"]
        terminal = input_["terminal"]
        bootstrap = input_["bootstrap"]
        seq_len = input_["seq_len"]
        err, _, _, _, _ = self.td_error(
            obs, hid, action, reward, terminal, bootstrap, seq_len
        )
        priority = err.abs()
        priority = self.aggregate_priority(priority, seq_len).detach().cpu()
        return {"priority": priority}

    def get_entropy(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
    ):
        priv_s = obs["priv_s"]
        assert (self.play_styles > 0)
        expand_playstyles = self.playstyle_list.unsqueeze(1).unsqueeze(0).expand(priv_s.shape[0],self.play_styles,priv_s.shape[1],self.play_styles)
        expand_hid = {}
        expand_hid['h0'] = hid['h0'].unsqueeze(1).expand(hid['h0'].shape[0],self.play_styles,*hid['h0'].shape[1:]).flatten(1, 2)
        expand_hid['c0'] = hid['c0'].unsqueeze(1).expand(hid['c0'].shape[0],self.play_styles,*hid['c0'].shape[1:]).flatten(1, 2)
        legal_move = obs["legal_move"]
        expand_priv_s = priv_s.unsqueeze(1).expand(priv_s.shape[0],self.play_styles,*priv_s.shape[1:])
        
        expand_legal_move = legal_move.unsqueeze(1).expand(legal_move.shape[0],self.play_styles,*legal_move.shape[1:]).flatten(1, 2)
        expand_obsinput = torch.cat((expand_priv_s,expand_playstyles),-1).flatten(1, 2)
        total_distribution = self.online_net.calculate_distribution(expand_obsinput,expand_legal_move,expand_hid)
        total_distribution = total_distribution.reshape(total_distribution.shape[0],self.play_styles,-1)
        mean_distribution = torch.mean(total_distribution,dim=0) + 1e-6

        entropy = torch.sum(-mean_distribution*torch.log(mean_distribution),dim=-1)

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        return entropy.sum(dim=0), entropy.mean().item()

    def get_real_mi(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ):
        priv_s = obs["priv_s"]
        assert (self.play_styles > 0)
        onehot_playstyle = nn.functional.one_hot(obs["playStyle"],num_classes=self.play_styles).float()
        assert priv_s.dim() == 3
        expand_hid = {}

        expand_hid['h0'] = hid['h0'].unsqueeze(1).expand(hid['h0'].shape[0],self.play_styles,*hid['h0'].shape[1:]).flatten(1, 2)
        expand_hid['c0'] = hid['c0'].unsqueeze(1).expand(hid['c0'].shape[0],self.play_styles,*hid['c0'].shape[1:]).flatten(1, 2)
        expanded_playstyles = self.playstyle_list.unsqueeze(1).unsqueeze(0).expand(onehot_playstyle.shape[0],self.play_styles,*onehot_playstyle.shape[1:])
        onehot_playstyle_expand = onehot_playstyle.unsqueeze(1).expand_as(expanded_playstyles)
        
        # priv_s = torch.cat((priv_s,obs["playStyle"]),dim=-1)
        legal_move = obs["legal_move"]
        
        expand_priv_s = priv_s.unsqueeze(1).expand(priv_s.shape[0],self.play_styles,*priv_s.shape[1:])
        expand_legal_move = legal_move.unsqueeze(1).expand(legal_move.shape[0],self.play_styles,*legal_move.shape[1:]).flatten(1, 2)
        expand_action = action.unsqueeze(1).expand(action.shape[0],self.play_styles,*action.shape[1:]).flatten(1, 2)
        expand_obsinput = torch.cat((expand_priv_s,self.generate_ps(expanded_playstyles)),-1).flatten(1, 2)
        act_a = self.online_net.calculate_maxval(expand_obsinput,expand_legal_move,expand_action,expand_hid)
        act_a = act_a.reshape(priv_s.shape[0],self.play_styles,-1)
        loss_mask = ((onehot_playstyle_expand != expanded_playstyles).sum(dim=-1) != 0).float()
        bin_loss = torch.mean(loss_mask*act_a,dim=1)
        extra_info = torch.mean((act_a>0).float()).item()

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        return bin_loss.sum(dim=0), extra_info

    def get_sim_mi(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ):
        priv_s = obs["priv_s"]
        assert (self.play_styles > 0)
        onehot_playstyle = nn.functional.one_hot(obs["playStyle"],num_classes=self.play_styles).float()
        onehot_playstyle_expand = onehot_playstyle.unsqueeze(1)
        assert priv_s.dim() == 3
        expand_hid = {}
        expand_hid['h0'] = hid['h0'].unsqueeze(1).expand(hid['h0'].shape[0],self.play_styles+1,*hid['h0'].shape[1:]).flatten(1, 2)
        expand_hid['c0'] = hid['c0'].unsqueeze(1).expand(hid['c0'].shape[0],self.play_styles+1,*hid['c0'].shape[1:]).flatten(1, 2)

        expanded_playstyles = self.playstyle_list.unsqueeze(1).unsqueeze(0).expand(onehot_playstyle.shape[0],self.play_styles,*onehot_playstyle.shape[1:])
        playstyle_encoding = torch.cat((onehot_playstyle_expand,self.generate_ps(expanded_playstyles)),dim=1).flatten(1, 2)
        
        # priv_s = torch.cat((priv_s,obs["playStyle"]),dim=-1)
        legal_move = obs["legal_move"]
        
        expand_priv_s = priv_s.unsqueeze(1).expand(priv_s.shape[0],self.play_styles+1,*priv_s.shape[1:]).flatten(1, 2)
        expand_obsinput = torch.cat((expand_priv_s,playstyle_encoding),-1)

        expand_legal_move = legal_move.unsqueeze(1).expand(legal_move.shape[0],self.play_styles+1,*legal_move.shape[1:]).flatten(1, 2)
        expand_action = action.unsqueeze(1).expand(action.shape[0],self.play_styles+1,*action.shape[1:]).flatten(1, 2)

        total_p_vals = self.online_net.calculate_p(expand_obsinput,expand_legal_move,expand_action,expand_hid)
        total_p_vals = total_p_vals.reshape(total_p_vals.shape[0],self.play_styles+1,-1)
        target_p_vals = total_p_vals[:,0,:]
        res_p_vals = torch.mean(total_p_vals[:,1:,:],dim=1)
        mutual_info = torch.log(target_p_vals/res_p_vals)
        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        return mutual_info.sum(dim=0), mutual_info.mean().item()

    @torch.jit.script_method
    def td_error(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        max_seq_len = obs["priv_s"].size(0)
        priv_s = obs["priv_s"]
        if self.play_styles > 0:
            onehot_playstyle = nn.functional.one_hot(obs["playStyle"],num_classes=self.play_styles).float()
            priv_s = torch.cat((priv_s,self.generate_ps(onehot_playstyle)),dim=-1)
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        action = action["a"]
        for k, v in hid.items():
            hid[k] = v.flatten(1, 2).contiguous()

        bsize, num_player = priv_s.size(1), 1
        if self.vdn:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)
            action = action.flatten(1, 2)

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qa, greedy_a, online_q, lstm_o = self.online_net(
            priv_s, publ_s, legal_move, action, hid
        )

        if self.off_belief:
            target = obs["target"]
        else:
            target_qa, _, target_q, _ = self.target_net(
                priv_s, publ_s, legal_move, greedy_a, hid
            )

            if self.boltzmann:
                temperature = obs["temperature"].flatten(1, 2).unsqueeze(2)
                # online_q: [seq_len, bathc * num_player, num_action]
                logit = online_q / temperature.clamp(min=1e-6)
                # logit: [seq_len, batch * num_player, num_action]
                legal_logit = logit - (1 - legal_move) * 1e30
                assert legal_logit.dim() == 3
                pa = nn.functional.softmax(legal_logit, 2).detach()
                # pa: [seq_len, batch * num_player, num_action]

                assert target_q.size() == pa.size()
                target_qa = (pa * target_q).sum(-1).detach()
                assert online_qa.size() == target_qa.size()

            if self.vdn:
                online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
                target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
                lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

            target_qa = torch.cat(
                [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
            )
            target_qa[-self.multi_step :] = 0
            assert target_qa.size() == reward.size()
            target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa

        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        if self.off_belief and "valid_fict" in obs:
            err = err * obs["valid_fict"]
        return err, lstm_o, online_q, greedy_a, hid

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, 5, 3)
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size
        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, 5, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aggregate_priority(self, priority, seq_len):
        p_mean = priority.sum(0) / seq_len
        p_max = priority.max(0)[0]
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority

    def loss(self, batch, aux_weight, stat, div_args={'weight':0}):
        err, lstm_o, online_q, greedy_a, hid = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()
        priority = self.aggregate_priority(priority, batch.seq_len).detach().cpu()

        loss = rl_loss
        extra_info = 0
        extra_loss = 0
        if div_args['weight'] > 0:
            if div_args['act_type'] == 0: 
                input_action = greedy_a
            else:
                input_action = batch.action["a"]
            if div_args['div_type'] == 0: # sim mi
                extra_loss, extra_info = self.get_sim_mi(batch.obs,input_action,hid)
                extra_loss = - extra_loss
            elif div_args['div_type'] == 1: # real mi
                extra_loss, extra_info = self.get_real_mi(batch.obs,input_action,hid)
            elif div_args['div_type'] == 2: # entropy
                extra_loss, extra_info = self.get_entropy(batch.obs,hid)
        
        if aux_weight <= 0:
            return loss, priority, online_q, extra_loss, extra_info

        if self.vdn:
            pred1 = self.aux_task_vdn(
                lstm_o,
                batch.obs["own_hand"],
                batch.obs["temperature"],
                batch.seq_len,
                rl_loss.size(),
                stat,
            )
            loss = rl_loss + aux_weight * pred1
        else:
            pred = self.aux_task_iql(
                lstm_o,
                batch.obs["own_hand"],
                batch.seq_len,
                rl_loss.size(),
                stat,
            )
            loss = rl_loss + aux_weight * pred

        return loss, priority, online_q, extra_loss, extra_info

    def behavior_clone_loss(self, online_q, batch, t, clone_bot, stat):
        max_seq_len = batch.obs["priv_s"].size(0)
        priv_s = batch.obs["priv_s"]
        if self.play_styles > 0:
            onehot_playstyle = nn.functional.one_hot(batch.obs["playStyle"],num_classes=self.play_styles).float()
            priv_s = torch.cat((priv_s,self.generate_ps(onehot_playstyle)),dim=-1)
        publ_s = batch.obs["publ_s"]
        legal_move = batch.obs["legal_move"]

        bsize, num_player = priv_s.size(1), 1
        if self.vdn:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)

        with torch.no_grad():
            target_logit, _ = clone_bot(priv_s, publ_s, None)
            target_logit = target_logit - (1 - legal_move) * 1e10
            target = nn.functional.softmax(target_logit, 2)

        logit = online_q / t
        # logit: [seq_len, batch * num_player, num_action]
        legal_logit = logit - (1 - legal_move) * 1e10
        log_distq = nn.functional.log_softmax(legal_logit, 2)

        assert log_distq.size() == target.size()
        assert log_distq.size() == legal_move.size()
        xent = -(target.detach() * log_distq).sum(2) / legal_move.sum(2).clamp(min=1e-3)
        if self.vdn:
            xent = xent.view(max_seq_len, bsize, num_player).sum(2)

        mask = torch.arange(0, max_seq_len, device=batch.seq_len.device)
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        assert xent.size() == mask.size()
        xent = xent * mask
        xent = xent.sum(0)
        stat["bc_loss"].feed(xent.mean().detach())
        return xent
