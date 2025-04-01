
from adarl.utils.buffers import ThDReplayBuffer
from adarl.utils.sb3_callbacks import EvalCallback_ep, SigintHaltCallback, PrintLrRunInfo, CheckpointCallbackRB
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from wandb.integration.sb3 import WandbCallback
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.session
import adarl.utils.utils
import inspect
import time
import torch as th
from rreal.algorithms.sac_helpers import build_vec_env, wrap_with_logger, EnvBuilderProtocol, VecEnvBuilderProtocol
from typing import Any


def build_eval_callbacks(eval_configurations : list[dict],
                         vec_env_builder : VecEnvBuilderProtocol,
                         run_folder : str,
                         base_seed : int ):
    callbacks = []
    for eval_conf in eval_configurations:
        ggLog.info(f"Building eval config '{eval_conf['name']}'")
        eval_env = vec_env_builder(env_builder_args=eval_conf["env_builder_args"],
                                    run_folder=run_folder+f"/eval_"+eval_conf["name"],
                                    seed=base_seed+100000000,
                                    num_envs=eval_conf["num_envs"],
                                    env_name=eval_conf["name"])
        callbacks.append(EvalCallback_ep(eval_env=eval_env,
                                    n_eval_episodes=eval_conf["eval_eps"],
                                    eval_freq_ep=eval_conf["eval_freq_ep"],
                                    deterministic=eval_conf["deterministic"]))
        ggLog.info(f"Built eval config '{eval_conf['name']}'")
    return callbacks

def solve_sb3(seed : int, folderName : str, run_id : str,
              args : dict[str,Any], 
              env_builder_args : dict[str,Any], 
              env_builder : EnvBuilderProtocol,
              vec_env_builder : VecEnvBuilderProtocol,
              collector_device : th.device = th.device("cpu"),
              debug_level = -2,
              no_wandb = False,
              eval_configurations : list[dict] = []):
    
    ggLog.info(f"Starting run")
    log_folder, session = adarl.utils.session.adarl_startup(inspect.getframeinfo(inspect.currentframe().f_back)[0],
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName,
                                                        debug=debug_level,
                                                        use_wandb=not no_wandb)

    th.cuda.set_sync_debug_mode("warn")
    device = adarl.utils.utils.torch_selectBestGpu()
    ggLog.info("Building env...")

    if vec_env_builder is None:
        vec_env_builder = lambda seed, run_folder, num_envs, env_builder_args, env_name="": build_vec_env(   env_builder=env_builder,
                                                                                                    env_builder_args=env_builder_args,
                                                                                                    log_folder=run_folder,
                                                                                                    seed=seed,
                                                                                                    num_envs=num_envs,
                                                                                                    collector_device=collector_device,
                                                                                                    env_action_device=collector_device)
    vec_env_builder = wrap_with_logger(vec_env_builder)
    
    parallel_envs = args["n_envs"]
    env = vec_env_builder(  seed=seed,
                            run_folder=log_folder,
                            num_envs=parallel_envs,
                            env_builder_args=env_builder_args)
    eval_env_builder_args = {}
    eval_env_builder_args.update(env_builder_args)
    env_builder_args["video_save_freq"] = 1

    ggLog.info("Built")

    if args["algorithm"].lower() == "sac":
        model = SAC("MultiInputPolicy", env, verbose=0,
                        batch_size=args["batch_size"],
                        buffer_size=10_000_000,
                        gamma=0.99,
                        learning_rate=args["lr"],
                        ent_coef="auto",
                        learning_starts=10_000,
                        tau=0.005,
                        gradient_steps=args["train_steps"],
                        train_freq=args["train_freq"],
                        target_entropy="auto",
                        seed = seed,
                        device=device,
                        policy_kwargs=dict(net_arch=[512,256]),
                        replay_buffer_class = ThDReplayBuffer,
                        replay_buffer_kwargs = {"storage_torch_device" : device},
                        tensorboard_log=folderName+f"/tensorboard")
    elif args["algorithm"].lower() == "ppo":    
        model = PPO("MultiInputPolicy", env, verbose=1,
                    n_steps = args["train_freq"],
                    batch_size=args["batch_size"],
                    learning_rate=args["lr"],
                    seed = seed,
                    device=device,
                    policy_kwargs=dict(net_arch=[512,256]),
                    # replay_buffer_class = ThDictReplayBuffer,
                    # replay_buffer_kwargs = {"storage_torch_device" : device},
                    tensorboard_log=folderName+f"/tensorboard")
    else:
        raise RuntimeError(f"Invalid algorithm '{args['algorithm']}.")


    callbacks = build_eval_callbacks(eval_configurations=eval_configurations,
                                        vec_env_builder=vec_env_builder,
                                        run_folder=log_folder,
                                        base_seed=seed)
    # callbacks.append(EvalCallback_ep(eval_env, best_model_save_path=log_folder+"/eval/EvalCallback",
    #                                         log_path=log_folder+"/eval/EvalCallback", eval_freq_ep=10,
    #                                         deterministic=True, render=False, verbose=True,
    #                                         n_eval_episodes = 1))
    # callbacks.append(EvalCallback_ep(eval_env, best_model_save_path=log_folder+"/eval100/EvalCallback",
    #                                         log_path=log_folder+"/eval100/EvalCallback", eval_freq_ep=100,
    #                                         deterministic=True, render=False, verbose=True,
    #                                         n_eval_episodes = 10))
    callbacks.append(CheckpointCallbackRB(save_freq_ep=100,
                                          save_best=True,
                                          save_path=log_folder+"/checkpoints",
                                          name_prefix="model_checkpoint",
                                          save_freq=None))
    callbacks.append(WandbCallback( model_save_path=f"{folderName}/wandb/save",
                                    verbose=2))
    callbacks.append(SigintHaltCallback())
    callbacks.append(PrintLrRunInfo(print_freq_ep=parallel_envs))
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=10_000_000,
                callback=callbacks)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")
    model.save(log_folder+"/trained_model")
