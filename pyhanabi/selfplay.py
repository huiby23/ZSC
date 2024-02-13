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
import time
import os
import sys
import argparse
import pprint
import pickle
import numpy as np
import torch
from torch import nn

from act_group import ActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--boltzmann_act", type=int, default=0)
    parser.add_argument("--min_t", type=float, default=1e-3)
    parser.add_argument("--max_t", type=float, default=1e-1)
    parser.add_argument("--num_t", type=int, default=80)
    parser.add_argument("--hide_action", type=int, default=0)
    parser.add_argument("--off_belief", type=int, default=0)
    parser.add_argument("--belief_model", type=str, default="None")
    parser.add_argument("--num_fict_sample", type=int, default=10)
    parser.add_argument("--belief_device", type=str, default="cuda:0")

    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--clone_bot", type=str, default="", help="behavior clone loss")
    parser.add_argument("--clone_weight", type=float, default=0.0)
    parser.add_argument("--clone_t", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument(
        "--eta", type=float, default=0.9, help="eta for aggregate priority"
    )
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/ffwd/lstm"
    )

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:0")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    # adversarial training setting
    parser.add_argument("--adv_type", type=int, default=0)
    parser.add_argument("--adv_ratio", type=float, default=0.0)   

    # non-parameter sharing setting
    parser.add_argument("--no_sharing", type=bool, default=False)     

    # playstyles setting
    parser.add_argument("--play_styles", type=int, default=0)
    parser.add_argument("--playstyle_embedding", type=int, default=0)
    parser.add_argument("--rand_perstep", type=bool, default=False)  

    # PBL-encoding training setting
    parser.add_argument("--training_type", type=int, default=0)
    # 0 - one main agent and one partner agent, training
    # 1 - partner vs partner training first, then main vs partner training 
    # 2 - one main agent and one partner agent, with each round randomly choosing main vs partner or partner vs partner
    parser.add_argument("--population_epoch", type=int, default=100) # used in type1 training
    parser.add_argument("--mutual_info_ratio", type=float, default=0)    
    parser.add_argument("--entropy_ratio", type=float, default=0)   

    # training setting
    args = parser.parse_args()
    if args.off_belief == 1:
        args.method = "iql"
        args.multi_step = 1
        assert args.net in ["publ-lstm"], "should only use publ-lstm style network"
        assert not args.shuffle_color
    assert args.method in ["vdn", "iql"]
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    expected_eps = np.mean(explore_eps)

    if args.boltzmann_act:
        boltzmann_beta = utils.generate_log_uniform(
            1 / args.max_t, 1 / args.min_t, args.num_t
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))
    else:
        boltzmann_t = []
        print("no boltzmann")

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.train_bomb,
        args.max_len,
    )

    dict_stats = {}
    dict_stats['score_mm'] = np.zeros(args.num_epoch)
    dict_stats['score_mp'] = np.zeros(args.num_epoch)
    dict_stats['score_pp'] = np.zeros(args.num_epoch)
    dict_stats['main_rl_loss'] = np.zeros(args.num_epoch)
    dict_stats['partner_rl_loss'] = np.zeros(args.num_epoch)
    dict_stats['partner_extra_loss'] = np.zeros(args.num_epoch)

    if args.no_sharing:
        agent = r2d2.R2D2Agent(
            (args.method == "vdn"),
            args.multi_step,
            args.gamma,
            args.eta,
            args.train_device,
            games[0].feature_size(args.sad),
            args.rnn_hid_dim,
            games[0].num_action(),
            args.net,
            args.num_lstm_layer,
            args.boltzmann_act,
            False,  # uniform priority
            args.off_belief,
            adv_type=args.adv_type,
            adv_ratio=args.adv_ratio,
        )
        agent.sync_target_with_online()      

        agent_p = r2d2.R2D2Agent(
            (args.method == "vdn"),
            args.multi_step,
            args.gamma,
            args.eta,
            args.train_device,
            games[0].feature_size(args.sad),
            args.rnn_hid_dim,
            games[0].num_action(),
            args.net,
            args.num_lstm_layer,
            args.boltzmann_act,
            False,  # uniform priority
            args.off_belief,
            adv_type=args.adv_type,
            adv_ratio=args.adv_ratio,
            play_styles=args.play_styles,
            play_style_embedding_dim=args.playstyle_embedding
        )
        agent_p.sync_target_with_online()  


        if args.load_model and args.load_model != "None":
            print("*****loading pretrained model*****")
            print(args.load_model)
            utils.load_weight(agent.online_net, (args.load_model+'model0.pthw'), args.train_device)
            utils.load_weight(agent_p.online_net, (args.load_model+'p_model0.pthw'), args.train_device)
            print("*****done*****")
        # partner model is saved along with the main model by default

        agent = agent.to(args.train_device)
        agent_p = agent_p.to(args.train_device)
        optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
        optim_p = torch.optim.Adam(agent_p.online_net.parameters(), lr=args.lr, eps=args.eps)
        print(agent)
        eval_agent = agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False, "adv_type": 0, "adv_ratio":0})
        eval_agent_p = agent_p.clone(args.train_device, {"vdn": False, "boltzmann_act": False, "adv_type": 0, "adv_ratio":0})

        replay_buffer = rela.RNNPrioritizedReplay(
            args.replay_buffer_size,
            args.seed,
            args.priority_exponent,
            args.priority_weight,
            args.prefetch,
        )
        replay_buffer_p = rela.RNNPrioritizedReplay(
            args.replay_buffer_size,
            args.seed,
            args.priority_exponent,
            args.priority_weight,
            args.prefetch,
        )
        belief_model = None
        
        print('Agent initialization complete. Disable parameter sharing.')
        agent_params = {'play_styles':args.play_styles, 'rand_perstep':args.rand_perstep}

        if args.training_type == 0: # regular training, only main and regular partner: 
            print('type 0 population based training')
            act_group = ActGroup(
                args.act_device,
                agent,
                args.seed,
                args.num_thread,
                args.num_game_per_thread,
                args.num_player,
                explore_eps,
                boltzmann_t,
                args.method,
                args.sad,
                args.shuffle_color,
                args.hide_action,
                True,  # trinary, 3 bits for aux task
                replay_buffer,
                args.multi_step,
                args.max_len,
                args.gamma,
                args.off_belief,
                belief_model,
                agent_p,
                replay_buffer_p,
                agent_params,
                split_train=False
            )

            context, threads = create_threads(
                args.num_thread,
                args.num_game_per_thread,
                act_group.actors,
                games,
            )

            act_group.start_nonsharing()
            context.start()
            while replay_buffer.size() < args.burn_in_frames:
                print("warming up replay buffer:", replay_buffer.size())
                time.sleep(2)

            print("Success, Done")
            print("=======================")

            frame_stat = dict()
            frame_stat["num_acts"] = 0
            frame_stat["num_buffer"] = 0

            stat = common_utils.MultiCounter(args.save_dir)
            tachometer = utils.Tachometer()
            stopwatch = common_utils.Stopwatch()

            for epoch in range(args.num_epoch):
                print("beginning of epoch: ", epoch)
                print(common_utils.get_mem_usage())
                tachometer.start()
                stat.reset()
                stopwatch.reset()
                main_loss_list = []
                raw_loss_list = []
                extra_loss_list = []
                for batch_idx in range(args.epoch_len):
                    num_update = batch_idx + epoch * args.epoch_len
                    if num_update % args.num_update_between_sync == 0:
                        agent.sync_target_with_online()
                        agent_p.sync_target_with_online()
                    if num_update % args.actor_sync_freq == 0:
                        act_group.update_model_nonsharing(agent,agent_p)

                    torch.cuda.synchronize()
                    stopwatch.time("sync and updating")

                    batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
                    stopwatch.time("sample data")

                    loss, priority, online_q, _ = agent.loss(batch, args.aux_weight, stat)
                    loss = (loss * weight).mean()
                    loss.backward()
                    main_loss_list.append(loss.item())
                    batch_p, weight_p = replay_buffer_p.sample(args.batchsize, args.train_device)
                    loss_p, priority_p, online_q_p, extra_loss = agent_p.loss(batch_p, args.aux_weight, stat,args.mutual_info_ratio,args.entropy_ratio)
                    loss_p = (loss_p * weight).mean()
                    extra_loss = (extra_loss * weight).mean()
                    raw_loss_list.append(loss_p.item())
                    extra_loss_list.append(extra_loss.item())
                    final_p_loss = loss_p - extra_loss
                    final_p_loss.backward()
                    torch.cuda.synchronize()
                    stopwatch.time("forward & backward")

                    g_norm = torch.nn.utils.clip_grad_norm_(
                        agent.online_net.parameters(), args.grad_clip
                    )
                    optim.step()
                    optim.zero_grad()
                    g_norm_p = torch.nn.utils.clip_grad_norm_(
                        agent_p.online_net.parameters(), args.grad_clip
                    )
                    optim_p.step()
                    optim_p.zero_grad()

                    torch.cuda.synchronize()
                    stopwatch.time("update model")

                    replay_buffer.update_priority(priority)
                    replay_buffer_p.update_priority(priority_p)
                    stopwatch.time("updating priority")

                    stat["loss"].feed(loss.detach().item())
                    stat["grad_norm"].feed(g_norm)
                    stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())
                    stat["loss_p"].feed(loss_p.detach().item())
                    stat["grad_norm_p"].feed(g_norm_p)
                    stat["boltzmann_t_p"].feed(batch_p.obs["temperature"][0].mean())

                count_factor = args.num_player if args.method == "vdn" else 1
                if epoch > 0 and epoch % 10 == 0:
                    print("EPOCH: %d" % epoch)
                    tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
                    stopwatch.summary()

                eval_seed = (9917 + epoch * 999999) % 7777777
                eval_agent.load_state_dict(agent.state_dict())
                eval_agent_p.load_state_dict(agent_p.state_dict())
                score_mm, perfect_mm, *_ = evaluate(
                    [eval_agent, eval_agent],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    device = args.act_device,
                )

                score_mp, perfect_mp, *_ = evaluate(
                    [eval_agent, eval_agent_p],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    params = [None,agent_params],
                    device = args.act_device,
                )

                score_pp, perfect_pp, *_ = evaluate(
                    [eval_agent_p, eval_agent_p],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    params = [agent_params,agent_params],
                    device = args.act_device,
                )

                dict_stats['score_mm'][epoch] = score_mm
                dict_stats['score_mp'][epoch] = score_mp
                dict_stats['score_pp'][epoch] = score_pp
                dict_stats['main_rl_loss'][epoch] = np.mean(main_loss_list)
                dict_stats['partner_rl_loss'][epoch] = np.mean(raw_loss_list)
                dict_stats['partner_extra_loss'][epoch] = np.mean(extra_loss_list)
                with open(os.path.join(args.save_dir, 'train_log.pkl'), 'wb') as pickle_file:
                    pickle.dump(dict_stats, pickle_file)


                force_save_name = None
                if epoch > 0 and epoch % 50 == 0:
                    force_save_name = "model_epoch%d" % epoch
                model_saved = saver.save(
                    None, agent.online_net.state_dict(), score_mp, False, force_save_name, agent_p.online_net.state_dict()
                )
                print(
                    "epoch %d, score_mm: %.4f, score_mp: %.4f, score_pp: %.4f, perfect_mm: %.2f, perfect_mp: %.2f, perfect_pp: %.2f"
                    % (epoch, score_mm, score_mp, score_pp, perfect_mm * 100, perfect_mp * 100, perfect_pp * 100)
                )
                print('main rl loss: %.3e, part rl loss: %.3e, part extra loss: %.3e'%(dict_stats['main_rl_loss'][epoch],dict_stats['partner_rl_loss'][epoch],dict_stats['partner_extra_loss'][epoch]))
                print("==========")
        elif args.training_type == 1: # train population first and then train main agents
            print('type1 training not supported for now')
        elif args.training_type == 2: # train population and main agents together
            print('type 2 population based training')
            act_group = ActGroup(
                args.act_device,
                agent,
                args.seed,
                args.num_thread,
                args.num_game_per_thread,
                args.num_player,
                explore_eps,
                boltzmann_t,
                args.method,
                args.sad,
                args.shuffle_color,
                args.hide_action,
                True,  # trinary, 3 bits for aux task
                replay_buffer,
                args.multi_step,
                args.max_len,
                args.gamma,
                args.off_belief,
                belief_model,
                agent_p,
                replay_buffer_p,
                agent_params,
                split_train=True,
            )

            context, threads = create_threads(
                args.num_thread,
                args.num_game_per_thread,
                act_group.actors,
                games,
            )

            act_group.start_nonsharing()
            context.start()
            while replay_buffer.size() < args.burn_in_frames:
                print("warming up replay buffer:", replay_buffer.size())
                time.sleep(2)

            print("Success, Done")
            print("=======================")

            frame_stat = dict()
            frame_stat["num_acts"] = 0
            frame_stat["num_buffer"] = 0

            stat = common_utils.MultiCounter(args.save_dir)
            stat_p = common_utils.MultiCounter(args.save_dir)
            tachometer = utils.Tachometer()
            stopwatch = common_utils.Stopwatch()

            for epoch in range(args.num_epoch):
                print("beginning of epoch: ", epoch)
                print(common_utils.get_mem_usage())
                tachometer.start()
                stat.reset()
                stat_p.reset()
                stopwatch.reset()
                main_loss_list = []
                raw_loss_list = []
                extra_loss_list = []
                for batch_idx in range(args.epoch_len):
                    num_update = batch_idx + epoch * args.epoch_len
                    if num_update % args.num_update_between_sync == 0:
                        agent.sync_target_with_online()
                        agent_p.sync_target_with_online()
                    if num_update % args.actor_sync_freq == 0:
                        act_group.update_model_nonsharing(agent,agent_p)

                    torch.cuda.synchronize()
                    stopwatch.time("sync and updating")

                    batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
                    stopwatch.time("sample data")

                    loss, priority, online_q, _ = agent.loss(batch, args.aux_weight, stat)
                    loss = (loss * weight).mean()
                    main_loss_list.append(loss.item())
                    loss.backward()
                    
                    batch_p, weight_p = replay_buffer_p.sample(args.batchsize, args.train_device)
                    loss_p, priority_p, online_q_p, extra_loss = agent_p.loss(batch_p, args.aux_weight, stat_p, args.mutual_info_ratio,args.entropy_ratio)
                    loss_p = (loss_p * weight).mean()
                    extra_loss = (extra_loss * weight).mean()
                    raw_loss_list.append(loss_p.item())
                    extra_loss_list.append(extra_loss.item())
                    final_p_loss = loss_p - extra_loss
                    final_p_loss.backward()
                    torch.cuda.synchronize()
                    stopwatch.time("forward & backward")

                    g_norm = torch.nn.utils.clip_grad_norm_(
                        agent.online_net.parameters(), args.grad_clip
                    )
                    optim.step()
                    optim.zero_grad()
                    g_norm_p = torch.nn.utils.clip_grad_norm_(
                        agent_p.online_net.parameters(), args.grad_clip
                    )
                    optim_p.step()
                    optim_p.zero_grad()

                    torch.cuda.synchronize()
                    stopwatch.time("update model")

                    replay_buffer.update_priority(priority)
                    replay_buffer_p.update_priority(priority_p)
                    stopwatch.time("updating priority")

                    stat["loss"].feed(loss.detach().item())
                    stat["grad_norm"].feed(g_norm)
                    stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())
                    stat["loss_p"].feed(loss_p.detach().item())
                    stat["grad_norm_p"].feed(g_norm_p)
                    stat["boltzmann_t_p"].feed(batch_p.obs["temperature"][0].mean())
                
                

                count_factor = args.num_player if args.method == "vdn" else 1
                if epoch > 0 and epoch % 10 == 0:
                    print("EPOCH: %d" % epoch)
                    tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
                    stopwatch.summary()

                eval_seed = (9917 + epoch * 999999) % 7777777
                eval_agent.load_state_dict(agent.state_dict())
                eval_agent_p.load_state_dict(agent_p.state_dict())

                score_mm, perfect_mm, *_ = evaluate(
                    [eval_agent, eval_agent],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    device = args.act_device,
                )

                score_mp, perfect_mp, *_ = evaluate(
                    [eval_agent, eval_agent_p],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    params = [None,agent_params],
                    device = args.act_device,
                )

                score_pp, perfect_pp, *_ = evaluate(
                    [eval_agent_p, eval_agent_p],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    params = [agent_params,agent_params],
                    device = args.act_device,
                )
                dict_stats['main_rl_loss'][epoch] = np.mean(main_loss_list)
                dict_stats['partner_rl_loss'][epoch] = np.mean(raw_loss_list)
                dict_stats['partner_extra_loss'][epoch] = np.mean(extra_loss_list)
                dict_stats['score_mm'][epoch] = score_mm
                dict_stats['score_mp'][epoch] = score_mp
                dict_stats['score_pp'][epoch] = score_pp
                with open(os.path.join(args.save_dir, 'train_log.pkl'), 'wb') as pickle_file:
                    pickle.dump(dict_stats, pickle_file)
                force_save_name = None
                if epoch > 0 and epoch % 50 == 0:
                    force_save_name = "model_epoch%d" % epoch
                model_saved = saver.save(
                    None, agent.online_net.state_dict(), score_mp, False, force_save_name, agent_p.online_net.state_dict()
                )
                print(
                    "epoch %d, score_mm: %.4f, score_mp: %.4f, score_pp: %.4f, perfect_mm: %.2f, perfect_mp: %.2f, perfect_pp: %.2f"
                    % (epoch, score_mm, score_mp, score_pp, perfect_mm * 100, perfect_mp * 100, perfect_pp * 100)
                )
                print('main rl loss: %.3e, part rl loss: %.3e, part extra loss: %.3e'%(dict_stats['main_rl_loss'][epoch],dict_stats['partner_rl_loss'][epoch],dict_stats['partner_extra_loss'][epoch]))
                print("==========")

    else:

        agent = r2d2.R2D2Agent(
            (args.method == "vdn"),
            args.multi_step,
            args.gamma,
            args.eta,
            args.train_device,
            games[0].feature_size(args.sad),
            args.rnn_hid_dim,
            games[0].num_action(),
            args.net,
            args.num_lstm_layer,
            args.boltzmann_act,
            False,  # uniform priority
            args.off_belief,
            adv_type=args.adv_type,
            adv_ratio=args.adv_ratio,
        )
        agent.sync_target_with_online()      

        if args.load_model and args.load_model != "None":
            if args.off_belief and args.belief_model != "None":
                belief_config = utils.get_train_config(args.belief_model)
                if args.load_model == '1':
                    args.load_model = belief_config["policy"]

            print("*****loading pretrained model*****")
            print(args.load_model)
            utils.load_weight(agent.online_net, args.load_model, args.train_device)
            print("*****done*****")

        # use clone bot for additional bc loss
        if args.clone_bot and args.clone_bot != "None":
            clone_bot = utils.load_supervised_agent(args.clone_bot, args.train_device)
        else:
            clone_bot = None

        agent = agent.to(args.train_device)
        optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
        print(agent)
        eval_agent = agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False, "adv_type": 0, "adv_ratio":0})

        replay_buffer = rela.RNNPrioritizedReplay(
            args.replay_buffer_size,
            args.seed,
            args.priority_exponent,
            args.priority_weight,
            args.prefetch,
        )

        belief_model = None
        if args.off_belief and args.belief_model != "None":
            print(f"load belief model from {args.belief_model}")
            from belief_model import ARBeliefModel

            belief_devices = args.belief_device.split(",")
            belief_config = utils.get_train_config(args.belief_model)
            belief_model = []
            for device in belief_devices:
                belief_model.append(
                    ARBeliefModel.load(
                        args.belief_model,
                        device,
                        5,
                        args.num_fict_sample,
                        belief_config["fc_only"],
                    )
                )

        act_group = ActGroup(
            args.act_device,
            agent,
            args.seed,
            args.num_thread,
            args.num_game_per_thread,
            args.num_player,
            explore_eps,
            boltzmann_t,
            args.method,
            args.sad,
            args.shuffle_color,
            args.hide_action,
            True,  # trinary, 3 bits for aux task
            replay_buffer,
            args.multi_step,
            args.max_len,
            args.gamma,
            args.off_belief,
            belief_model,
        )

        context, threads = create_threads(
            args.num_thread,
            args.num_game_per_thread,
            act_group.actors,
            games,
        )

        act_group.start()
        context.start()
        while replay_buffer.size() < args.burn_in_frames:
            print("warming up replay buffer:", replay_buffer.size())
            time.sleep(1)

        print("Success, Done")
        print("=======================")

        frame_stat = dict()
        frame_stat["num_acts"] = 0
        frame_stat["num_buffer"] = 0

        stat = common_utils.MultiCounter(args.save_dir)
        tachometer = utils.Tachometer()
        stopwatch = common_utils.Stopwatch()

        for epoch in range(args.num_epoch):
            print("beginning of epoch: ", epoch)
            print(common_utils.get_mem_usage())
            tachometer.start()
            stat.reset()
            stopwatch.reset()

            for batch_idx in range(args.epoch_len):
                num_update = batch_idx + epoch * args.epoch_len
                if num_update % args.num_update_between_sync == 0:
                    agent.sync_target_with_online()
                if num_update % args.actor_sync_freq == 0:
                    act_group.update_model(agent)

                torch.cuda.synchronize()
                stopwatch.time("sync and updating")

                batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
                stopwatch.time("sample data")

                loss, priority, online_q, _ = agent.loss(batch, args.aux_weight, stat)
                if clone_bot is not None and args.clone_weight > 0:
                    bc_loss = agent.behavior_clone_loss(
                        online_q, batch, args.clone_t, clone_bot, stat
                    )
                    loss = loss + bc_loss * args.clone_weight
                loss = (loss * weight).mean()
                loss.backward()

                torch.cuda.synchronize()
                stopwatch.time("forward & backward")

                g_norm = torch.nn.utils.clip_grad_norm_(
                    agent.online_net.parameters(), args.grad_clip
                )
                optim.step()
                optim.zero_grad()

                torch.cuda.synchronize()
                stopwatch.time("update model")

                replay_buffer.update_priority(priority)
                stopwatch.time("updating priority")

                stat["loss"].feed(loss.detach().item())
                stat["grad_norm"].feed(g_norm)
                stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

            count_factor = args.num_player if args.method == "vdn" else 1
            print("EPOCH: %d" % epoch)
            tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
            stopwatch.summary()

            eval_seed = (9917 + epoch * 999999) % 7777777
            eval_agent.load_state_dict(agent.state_dict())
            score, perfect, *_ = evaluate(
                [eval_agent for _ in range(args.num_player)],
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
                device = args.act_device,
            )
            dict_stats['score_mm'][epoch] = score
            with open(os.path.join(args.save_dir, 'train_log.pkl'), 'wb') as pickle_file:
                pickle.dump(dict_stats, pickle_file)
            force_save_name = None
            if epoch > 0 and epoch % 50 == 0:
                force_save_name = "model_epoch%d" % epoch
            model_saved = saver.save(
                None, agent.online_net.state_dict(), score, force_save_name=force_save_name
            )
            print(
                "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
                % (epoch, score, perfect * 100, model_saved)
            )

            if clone_bot is not None:
                score, perfect, *_ = evaluate(
                    [clone_bot] + [eval_agent for _ in range(args.num_player - 1)],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    device = args.act_device,
                )
                print(f"clone bot score: {np.mean(score)}")

            if args.off_belief:
                actors = common_utils.flatten(act_group.actors)
                success_fict = [actor.get_success_fict_rate() for actor in actors]
                print(
                    "epoch %d, success rate for sampling ficticious state: %.2f%%"
                    % (epoch, 100 * np.mean(success_fict))
                )
            print("==========")
