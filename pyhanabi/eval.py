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
                    runners[2*i], num_player, i, False, sad[2*i], hide_action[2*i]
                )
                partner_actor = hanalearn.R2D2Actor(
                    runners[2*i+1], num_player, i, False, sad[2*i+1], hide_action[2*i+1], True
                )
                actors.append(main_actor)
                actors.append(partner_actor)

            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        record_name = game_name + "_" + str(t_idx)
        record_name_set.append(record_name)
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True, 1, record_name)
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
    return scores, num_perfect / len(scores), scores, num_perfect, all_actors

def evaluate_and_record_three(
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
    assert len(agents) == 5

    num_player = 3
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
            
            actor_a = hanalearn.R2D2Actor(runners[0], 3, 0, False, sad[0], hide_action[0])
            actor_b = hanalearn.R2D2Actor(runners[1], 3, 1, False, sad[0], hide_action[0])
            actor_b_ref = hanalearn.R2D2Actor(runners[2], 3, 1, False, sad[0], hide_action[0], True)
            actor_c = hanalearn.R2D2Actor(runners[3], 3, 2, False, sad[1], hide_action[1])
            actor_c_ref = hanalearn.R2D2Actor(runners[4], 3, 2, False, sad[0], hide_action[0], True)

            actors = [actor_a,actor_b,actor_b_ref, actor_c,actor_c_ref]
            thread_actors.append(actors)
            thread_games.append(games[g_idx])
        record_name = game_name + "_" + str(t_idx)
        record_name_set.append(record_name)
        thread = hanalearn.HanabiThreadLoop(thread_games, thread_actors, True, 2, record_name)
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
    return scores, num_perfect / len(scores), scores, num_perfect, all_actors


def get_similarities(record_name, thread_num=80):
    agent_a_sim = 0
    agent_a_diff = 0
    agent_b_sim = 0
    agent_b_diff = 0    
    for idx in range(thread_num):
        full_record_name = "records/"+record_name+"_"+str(idx)
        thread_results = np.loadtxt(full_record_name+".txt")
        agent_a_sim += thread_results[0]
        agent_a_diff += thread_results[1]
        agent_b_sim += thread_results[2]
        agent_b_diff += thread_results[3]

    return  agent_a_sim/(agent_a_sim+agent_a_diff), agent_b_sim/(agent_b_sim+agent_b_diff)

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
    params=[None,None],
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
                if params[i]:
                    actor = hanalearn.R2D2Actor(
                        runners[i], num_player, i, False, sad[i], hide_action[i], params[i]["play_styles"], params[i]["encoding_duplicate"], params[i]["rand_perstep"]
                    )
                else:
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


def evaluate_saved_model_three(
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

    for i in range(num_run):
        _, _, score, p, games = evaluate_and_record_three(
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
        for i in range(num_run):
            _, _, score, p, games = evaluate_and_record(
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
            scores.extend(score)
            perfect += p
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
