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
import argparse
import os
import sys
import pprint
import itertools
from collections import defaultdict
import numpy as np


lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
import common_utils
import utils
from tools import model_zoo


def filter_include(entries, includes):
    if not isinstance(includes, list):
        includes = [includes]
    keep = []
    for entry in entries:
        for include in includes:
            if include not in entry:
                break
        else:
            keep.append(entry)
    return keep


def filter_exclude(entries, excludes):
    if not isinstance(excludes, list):
        excludes = [excludes]
    keep = []
    for entry in entries:
        for exclude in excludes:
            if exclude in entry:
                break
        else:
            keep.append(entry)
    return keep


def cross_play(models, num_player, num_game, seed, device, record_name=None):
    if args.root is not None:
        combs = list(itertools.combinations_with_replacement(models, num_player))
    else:
        combs = [models]
    perfs = defaultdict(list)
    for comb in combs:
        num_model = len(set(comb))
        score = evaluate_saved_model(comb, num_game, seed, 0, device=device, record_name=record_name)[0]
        perfs[num_model].append(score)
    if args.root is not None:
        for num_model, scores in perfs.items():
            print(
                f"#model: {num_model}, #groups {len(scores)}, "
                f"score: {np.mean(scores):.2f} "
                f"+/- {np.std(scores) / np.sqrt(len(scores) - 1):.2f}"
            )


parser = argparse.ArgumentParser()
parser.add_argument("--root", default=None, type=str)
parser.add_argument("--num_player", default=None, type=int, required=False)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--include", default=None, type=str, nargs="+")
parser.add_argument("--exclude", default=None, type=str, nargs="+")
parser.add_argument("--model_a", default=None, type=str)
parser.add_argument("--model_b", default=None, type=str)
parser.add_argument("--model_ap", default=None, type=str)
parser.add_argument("--model_bp", default=None, type=str)
parser.add_argument("--record_name", default=None, type=str)

args = parser.parse_args()

if args.root is not None:
    models = common_utils.get_all_files(args.root, "model0.pthw")
    if args.include is not None:
        models = filter_include(models, args.include)
    if args.exclude is not None:
        models = filter_exclude(models, args.exclude)

    pprint.pprint(models)
    cross_play(models, args.num_player, 1000, 1, args.device, args.record_name)
else:
    if args.record_name is not None:
        models = [args.model_a, args.model_bp, args.model_b, args.model_ap]
    else:
        models = [args.model_a, args.model_b]
    cross_play(models, args.num_player, 1000, 1, args.device, args.record_name)
