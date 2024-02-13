
# PBL training type 1 
print('type 1 population based training')
print('stage 1 training: train a population')
act_group_p = ActGroup(
    args.act_device,
    agent_p,
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
    replay_buffer_p,
    args.multi_step,
    args.max_len,
    args.gamma,
    args.off_belief,
    belief_model,
    agent_params = agent_params
)

context, threads = create_threads(
    args.num_thread,
    args.num_game_per_thread,
    act_group_p.actors,
    games,
)

act_group_p.start()
context.start()

while replay_buffer_p.size() < args.burn_in_frames:
    print("warming up replay buffer:", replay_buffer_p.size())
    time.sleep(2)

print("Success, Done")
print("=======================")

frame_stat = dict()
frame_stat["num_acts"] = 0
frame_stat["num_buffer"] = 0

stat = common_utils.MultiCounter(args.save_dir)
tachometer = utils.Tachometer()
stopwatch = common_utils.Stopwatch()

for epoch in range(args.population_epoch):
    print("beginning of epoch: ", epoch)
    print(common_utils.get_mem_usage())
    tachometer.start()
    stat.reset()
    stopwatch.reset()

    for batch_idx in range(args.epoch_len):
        num_update = batch_idx + epoch * args.epoch_len
        if num_update % args.num_update_between_sync == 0:
            agent_p.sync_target_with_online()
        if num_update % args.actor_sync_freq == 0:
            act_group_p.update_model(agent_p)

        torch.cuda.synchronize()
        stopwatch.time("sync and updating")

        batch, weight = replay_buffer_p.sample(args.batchsize, args.train_device)
        stopwatch.time("sample data")
        loss, priority, online_q, extra_loss = agent_p.loss(batch, args.aux_weight, stat,args.mutual_info_ratio,args.entropy_ratio)
        loss = (loss * weight).mean()
        extra_loss = (extra_loss * weight).mean()
        final_p_loss = loss - extra_loss
        final_p_loss.backward()
        torch.cuda.synchronize()
        stopwatch.time("forward & backward")

        g_norm = torch.nn.utils.clip_grad_norm_(
            agent_p.online_net.parameters(), args.grad_clip
        )
        optim.step()
        optim.zero_grad()

        torch.cuda.synchronize()
        stopwatch.time("update model")

        replay_buffer_p.update_priority(priority)
        stopwatch.time("updating priority")

        stat["loss"].feed(loss.detach().item())
        stat["grad_norm"].feed(g_norm)
        stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

    count_factor = args.num_player if args.method == "vdn" else 1
    if epoch > 0 and epoch % 10 == 0:
        print("EPOCH: %d" % epoch)
        tachometer.lap(replay_buffer_p, args.epoch_len * args.batchsize, count_factor)
        stopwatch.summary()

    eval_seed = (9917 + epoch * 999999) % 7777777
    eval_agent_p.load_state_dict(agent_p.state_dict())
    score, perfect, *_ = evaluate(
        [eval_agent_p for _ in range(args.num_player)],
        1000,
        eval_seed,
        args.eval_bomb,
        0,  # explore eps
        args.sad,
        args.hide_action,
        params = [agent_params,agent_params],
        device = args.act_device,
    )
    dict_stats['score_pp'][epoch] = score
    with open(os.path.join(args.save_dir, 'train_log.pkl'), 'wb') as pickle_file:
        pickle.dump(dict_stats, pickle_file)
    force_save_name = "partner_model"
    if epoch > 0 and epoch % 50 == 0:
        force_save_name = "partner_model_epoch%d" % epoch
    model_saved = saver.save(
        None, agent_p.online_net.state_dict(), score, force_save_name=force_save_name
    )
    print(
        "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
        % (epoch, score, perfect * 100, model_saved)
    )

print('stage 1 training complete, start stage 2 training')
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
    agent_params
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

for epoch in range(args.population_epoch, args.num_epoch):
    print("beginning of epoch: ", epoch)
    print(common_utils.get_mem_usage())
    tachometer.start()
    stat.reset()
    stopwatch.reset()

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

        # in stage 2 training, we do not update partner parameters                    
        # batch_p, weight_p = replay_buffer_p.sample(args.batchsize, args.train_device)
        # loss_p, priority_p, online_q_p = agent_p.loss(batch_p, args.aux_weight, stat)
        # loss_p = (loss_p * weight).mean()
        # loss_p.backward()
        torch.cuda.synchronize()
        stopwatch.time("forward & backward")

        g_norm = torch.nn.utils.clip_grad_norm_(
            agent.online_net.parameters(), args.grad_clip
        )
        optim.step()
        optim.zero_grad()
        # g_norm_p = torch.nn.utils.clip_grad_norm_(
        #     agent_p.online_net.parameters(), args.grad_clip
        # )
        # optim_p.step()
        # optim_p.zero_grad()

        torch.cuda.synchronize()
        stopwatch.time("update model")

        replay_buffer.update_priority(priority)
        # replay_buffer_p.update_priority(priority_p)
        stopwatch.time("updating priority")

        stat["loss"].feed(loss.detach().item())
        stat["grad_norm"].feed(g_norm)
        stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())
        # stat["loss_p"].feed(loss_p.detach().item())
        # stat["grad_norm_p"].feed(g_norm_p)
        # stat["boltzmann_t_p"].feed(batch_p.obs["temperature"][0].mean())

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
    dict_stats['score_mm'][epoch] = score_mm
    dict_stats['score_mp'][epoch] = score_mp
    with open(os.path.join(args.save_dir, 'train_log.pkl'), 'wb') as pickle_file:
        pickle.dump(dict_stats, pickle_file)
    force_save_name = None
    if epoch > 0 and epoch % 50 == 0:
        force_save_name = "model_epoch%d" % epoch
    model_saved = saver.save(
        None, agent.online_net.state_dict(), score_mp, False, force_save_name, agent_p.online_net.state_dict()
    )
    print(
        "epoch %d, score_mm: %.4f, score_mp: %.4f, perfect_mm: %.2f, perfect_mp: %.2f"
        % (epoch, score_mm, score_mp, perfect_mm * 100, perfect_mp * 100)
    )
    print("==========")