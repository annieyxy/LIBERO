import os
from pathlib import Path
import numpy as np
import cv2
import torch
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import bddl_utils as BDDLUtils
from libero.libero import benchmark, get_libero_path


benchmark_name = "libero_90"
task_ids = [i for i in range(90)] # list of task indices in the suite
max_init_states = 1  # per task
camera_height = 256
camera_width = 256
settle_steps = 5
output_dir = Path("rendered_pairs")
output_dir.mkdir(parents=True, exist_ok=True)

# Set True to also keep objects that appear in goals (in addition to obj_of_interest)
keep_goal_objects = True


def _body_id2name(model, body_id):
    if hasattr(model, "body_id2name"):
        return model.body_id2name(body_id)
    if hasattr(model, "body_names"):
        name = model.body_names[body_id]
        return name.decode() if isinstance(name, (bytes, bytearray)) else name
    # Fallback: try name2id map if available
    if hasattr(model, "body_names"):
        return model.body_names[body_id]
    raise RuntimeError("Unable to resolve body names from mujoco model")


def _collect_goal_objects(parsed_problem):
    goal_objects = set()
    for state in parsed_problem.get("goal_state", []):
        if len(state) >= 2:
            goal_objects.add(state[1])
        if len(state) >= 3:
            goal_objects.add(state[2])
    return goal_objects


def _collect_all_objects_in_tasks(bench, task_ids):
    all_objects = set()
    for task_id in task_ids:
        bddl_path = bench.get_task_bddl_file_path(task_id)
        parsed = BDDLUtils.robosuite_parse_problem(bddl_path)
        for obj_list in parsed.get("objects", {}).values():
            all_objects.update(obj_list)
    return all_objects


def set_object_alpha(env, object_names, alpha=0.0):
    model = env.sim.model
    object_names = set(object_names)
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = _body_id2name(model, body_id)
        if body_name is None:
            continue
        for obj_name in object_names:
            if body_name.startswith(obj_name):
                model.geom_rgba[geom_id][3] = alpha
                break


def hide_distractors(env, keep_objects):
    all_objects = set(env.env.objects_dict.keys())
    to_hide = [name for name in all_objects if name not in keep_objects]
    if to_hide:
        set_object_alpha(env, to_hide, alpha=0.0)
    return to_hide


def render_from_state(env, init_state, settle_steps=5):
    obs = env.set_init_state(init_state)
    for _ in range(settle_steps):
        obs, _, _, _ = env.step([0.0] * 7)
    return obs["agentview_image"]



# Build benchmark and preview tasks
bench = benchmark.get_benchmark_dict()[benchmark_name]()
print("Tasks:")
for i in task_ids:
    task = bench.get_task(i)
    print(i, task.name, "|", task.language)

all_objects_in_suite = _collect_all_objects_in_tasks(bench, task_ids)


# generate pairs
for task_id in task_ids:
    task = bench.get_task(task_id)
    bddl_path = bench.get_task_bddl_file_path(task_id)
    init_states = bench.get_task_init_states(task_id)
    if max_init_states is not None:
        init_states = init_states[:max_init_states]

    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": camera_height,
        "camera_widths": camera_width,
    }

    env_clutter = OffScreenRenderEnv(**env_args)
    env_clean = OffScreenRenderEnv(**env_args)
    env_clutter.reset()
    env_clean.reset()

    keep_objects = set(env_clean.obj_of_interest)
    if keep_goal_objects:
        keep_objects |= _collect_goal_objects(env_clean.env.parsed_problem)

    for idx, state in enumerate(init_states):
        scene_objects = set(env_clean.env.objects_dict.keys())
        missing_candidates = sorted(all_objects_in_suite - scene_objects)
        missing_object = (
            np.random.choice(missing_candidates) if missing_candidates else "none"
        )

        # cluttered
        img_clutter = render_from_state(env_clutter, state, settle_steps=settle_steps)

        # clean (hide distractors)
        obs = env_clean.set_init_state(state)
        hidden = hide_distractors(env_clean, keep_objects=keep_objects)
        for _ in range(settle_steps):
            obs, _, _, _ = env_clean.step([0.0] * 7)
        img_clean = obs["agentview_image"]

        base = f"{task.name}_state{idx:04d}"
        clutter_path = output_dir / f"{base}_clutter_no_{missing_object}.png"
        clean_path = output_dir / f"{base}_clean_no_{missing_object}.png"

        cv2.imwrite(str(clutter_path), img_clutter[::-1, :, ::-1])
        cv2.imwrite(str(clean_path), img_clean[::-1, :, ::-1])

        if idx == 0:
            print(f"Task {task_id} hidden objects: {hidden}")

    env_clutter.close()
    env_clean.close()

print(f"Saved pairs to: {output_dir}")


# multiple of same objects - shows up as diff object
