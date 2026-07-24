import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import numpy as np
from argparse import ArgumentParser
from envs._base_task import Base_Task
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, sapien.Pose):
        return value.p.tolist() + value.q.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def append_failure_record(
    args,
    seed,
    failure_type,
    message,
    details=None,
    episode_index=None,
    phase="seed_collection",
):
    try:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_name": args["task_name"],
            "task_config": args["task_config"],
            "phase": phase,
            "seed": int(seed),
            "episode_index": episode_index,
            "failure_type": failure_type,
            "message": str(message),
            "details": _json_safe(details or {}),
        }
        log_path = os.path.join(args["save_path"], "failure_seeds.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as log_error:
        print(f"Warning: failed to write failure seed log: {log_error}")


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]) + ", save video: " +
          str(args["camera"].get("save_wrist_camera_video", False)))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV: Base_Task, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    collection_seed_reserve = (
        max(0, int(args.get("collection_seed_reserve", 0)))
        if args.get("collect_data", False)
        else 0
    )
    seed_target_num = args["episode_num"] + collection_seed_reserve

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        if collection_seed_reserve:
            print(
                f"Collecting {collection_seed_reserve} extra seed(s) "
                "for failed trajectory replays."
            )

        while suc_num < seed_target_num:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                task_success = TASK_ENV.check_success() if TASK_ENV.plan_success else False
                if TASK_ENV.plan_success and task_success:
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    details = getattr(TASK_ENV, "last_failure_info", None)
                    failure_type = "planning_failed" if not TASK_ENV.plan_success else "task_check_failed"
                    append_failure_record(
                        args,
                        epid,
                        failure_type,
                        "Planning failed" if not TASK_ENV.plan_success else "Task success check failed",
                        details=details,
                        episode_index=suc_num,
                    )
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                append_failure_record(
                    args,
                    epid,
                    "objects_unstable",
                    e,
                    details=getattr(TASK_ENV, "last_failure_info", None),
                    episode_index=suc_num,
                )
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                details = getattr(TASK_ENV, "last_failure_info", None)
                failure_type = (details.get("failure_type", "exception")
                                if isinstance(details, dict) else "exception")
                append_failure_record(
                    args,
                    epid,
                    failure_type,
                    e,
                    details=details,
                    episode_index=suc_num,
                )
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        manifest_path = os.path.join(args["save_path"], "collection_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as file:
                collection_manifest = json.load(file)
        else:
            collection_manifest = {}

        mapped_candidate_indices = [
            int(item["trajectory_index"])
            for item in collection_manifest.values()
            if isinstance(item, dict) and "trajectory_index" in item
        ]
        candidate_idx = (
            max(mapped_candidate_indices) + 1
            if mapped_candidate_indices
            else st_idx
        )
        episode_idx = st_idx

        while episode_idx < args["episode_num"]:
            if candidate_idx >= len(seed_list):
                raise RuntimeError(
                    "Not enough replayable seed trajectories to finish data collection. "
                    "Increase 'collection_seed_reserve' in the task config and rerun."
                )

            seed = seed_list[candidate_idx]
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **args)

            traj_data = TASK_ENV.load_tran_data(candidate_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            collection_success = TASK_ENV.plan_success and TASK_ENV.check_success()

            if not collection_success:
                success_check = getattr(TASK_ENV, "last_success_check", None)
                print(" -------------")
                print(
                    f"collect data episode {episode_idx} fail! "
                    f"(seed = {seed}, trajectory = {candidate_idx})"
                )
                print("Success check: ", success_check)
                print("Skip this trajectory and try the next saved seed.")
                print(" -------------")
                append_failure_record(
                    args,
                    seed,
                    "collection_replay_failed",
                    "Saved trajectory replay did not satisfy the task success check",
                    details={
                        "trajectory_index": candidate_idx,
                        "success_check": success_check,
                        "last_failure_info": getattr(TASK_ENV, "last_failure_info", None),
                    },
                    episode_index=episode_idx,
                    phase="data_collection",
                )
                TASK_ENV.close_env()
                TASK_ENV.remove_data_cache()
                candidate_idx += 1
                continue

            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()

            collection_manifest[f"episode_{episode_idx}"] = {
                "seed": int(seed),
                "trajectory_index": int(candidate_idx),
            }
            with open(manifest_path, "w", encoding="utf-8") as file:
                json.dump(collection_manifest, file, ensure_ascii=False, indent=4)

            episode_idx += 1
            candidate_idx += 1

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
