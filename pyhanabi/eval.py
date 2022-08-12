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
import os
import time
import json
import numpy as np
import torch

from create import *
import rela
import r2d2
import utils
import warnings

def evaluate_and_record(
    agents,
    num_game,
    seed,
    bomb,
    eps,
    sad,
    hide_action,
    *,
    num_thread=10,
    max_len=80,
    device="cuda:0",
    game_name="iql_1_vs_iql_2",
):
    """
    evaluate agents as long as they have a "act" function
    """
    if num_game < num_thread:
        num_thread = num_game
    assert len(agents) == 4

    num_player = 2
    if not isinstance(hide_action, list):
        hide_action = [hide_action for _ in range(num_player)]
    if not isinstance(sad, list):
        sad = [sad for _ in range(num_player)]
    runners = [rela.BatchRunner(agent, device, 1000, ["act"]) for agent in agents]

    context = rela.Context()
    games = create_envs(num_game, seed, num_player, bomb, max_len)
    threads = []

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []
    record_name_set = []
    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                main_actor = hanalearn.R2D2Actor(
                    runners[2*i], num_player, i, False, sad[i], hide_action[i]
                )
                partner_actor = hanalearn.R2D2Actor(
                    runners[2*i+1], num_player, i, False, sad[i], hide_action[i], True
                )
                actors.append(main_actor)
                actors.append(partner_actor)

            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        record_name = game_name + "_" + str(t_idx)
        record_name_set.append(record_name)
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True, True, record_name)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    main_1_actions = []
    partner_1_actions = []
    main_2_actions = []
    partner_2_actions = []
    for record_name in record_name_set:
        print('record name:', record_name)
        main_1_action = np.loadtxt("../templogs/"+record_name+"_1m.txt")
        print("load from:", "../templogs/"+record_name+"_1m.txt")
        print("main_shape:", main_1_action.shape)
        partner_1_action = np.loadtxt("../templogs/"+record_name+"_1p.txt")
        print("partner_shape:", partner_1_action.shape)
        main_2_action = np.loadtxt("../templogs/"+record_name+"_2m.txt")
        partner_2_action = np.loadtxt("../templogs/"+record_name+"_2p.txt")
        main_1_actions.append(main_1_action) 
        partner_1_actions.append(partner_1_action)
        main_2_actions.append(main_2_action)
        partner_2_actions.append(partner_2_action)
            
        os.remove("../templogs/"+record_name+"_1m.txt")
        os.remove("../templogs/"+record_name+"_1p.txt")
        os.remove("../templogs/"+record_name+"_2m.txt")
        os.remove("../templogs/"+record_name+"_2p.txt")
    
    main_1_actions_total = np.hstack(main_1_actions)
    partner_1_actions_total = np.hstack(partner_1_actions)
    main_2_actions_total = np.hstack(main_2_actions)
    partner_2_actions_total = np.hstack(partner_2_actions)

    run_record_1 = np.vstack([main_1_actions_total, partner_1_actions_total])
    run_record_2 = np.vstack([main_2_actions_total, partner_2_actions_total])

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return run_record_1,run_record_2, num_perfect / len(scores), scores, num_perfect, all_actors

def evaluate(
    agents,
    num_game,
    seed,
    bomb,
    eps,
    sad,
    hide_action,
    *,
    num_thread=10,
    max_len=80,
    device="cuda:0",
):
    """
    evaluate agents as long as they have a "act" function
    """
    if num_game < num_thread:
        num_thread = num_game

    num_player = len(agents)
    if not isinstance(hide_action, list):
        hide_action = [hide_action for _ in range(num_player)]
    if not isinstance(sad, list):
        sad = [sad for _ in range(num_player)]
    runners = [rela.BatchRunner(agent, device, 1000, ["act"]) for agent in agents]

    context = rela.Context()
    games = create_envs(num_game, seed, num_player, bomb, max_len)
    threads = []

    assert num_game % num_thread == 0
    game_per_thread = num_game // num_thread
    all_actors = []
    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                actor = hanalearn.R2D2Actor(
                    runners[i], num_player, i, False, sad[i], hide_action[i]
                )
                actors.append(actor)
                all_actors.append(actor)
            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    scores = [g.last_episode_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect, all_actors


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    overwrite=None,
    num_run=1,
    verbose=True,
    record_name=None,
    device="cuda:0",
):
    agents = []
    sad = []
    hide_action = []
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = device
    overwrite["boltzmann_act"] = False

    for weight_file in weight_files:
        state_dict = torch.load(weight_file, map_location=device)
        if "fc_v.weight" in state_dict.keys():
            agent, cfg = utils.load_agent(weight_file, overwrite)
            agents.append(agent)
            sad.append(cfg["sad"] if "sad" in cfg else cfg["greedy_extra"])
            hide_action.append(bool(cfg["hide_action"]))
        else:
            agent = utils.load_supervised_agent(weight_file, device)
            agents.append(agent)
            sad.append(False)
            hide_action.append(False)
        agent.train(False)

    scores = []
    perfect = 0
    if record_name is not None:
        total_record_1 = []
        total_record_2 = []
        for i in range(num_run):
            run_record_1, run_record_2, _, score, p, games = evaluate_and_record(
                agents,
                num_game,
                num_game * i + seed,
                bomb,
                0,  # eps
                sad,
                hide_action,
                game_name = record_name,
                device = device,
            )
            total_record_1.append(run_record_1)
            total_record_2.append(run_record_2)
            scores.extend(score)
            perfect += p
        if num_run > 1:
            record_data_1 = np.hstack(total_record_1)
            record_data_2 = np.hstack(total_record_2)
        else:
            record_data_1 = total_record_1[0]
            record_data_2 = total_record_2[0]
        np.save("../dataset/"+record_name+"_a.npy",record_data_1)
        np.save("../dataset/"+record_name+"_b.npy",record_data_2)
        print("data saved:", record_name)

    else:
        for i in range(num_run):
            _, _, score, p, games = evaluate(
                agents,
                num_game,
                num_game * i + seed,
                bomb,
                0,  # eps
                sad,
                hide_action,
            )
            scores.extend(score)
            perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print(
            "score: %.3f +/- %.3f" % (mean, sem),
            "; perfect: %.2f%%" % (100 * perfect_rate),
        )
    return mean, sem, perfect_rate, scores, games
