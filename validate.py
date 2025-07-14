import env
import PPO_model as PPO_model
import torch
import time
import os
import copy
import gym

def get_validate_env(env_paras):
    """
    Load all .fjs instance files from data_dev/<jobs><machines:02d>/ 
    and build a validation env whose batch_size matches the number of files.
    """
    # 1) Build subdir name (e.g. "503")
    subdir = f"{env_paras['num_jobs']}{env_paras['num_mas']:02d}"
    dirpath = os.path.join("data_dev", subdir)
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Validation directory not found: {dirpath}")

    # 2) Collect all non-empty .fjs files
    case_files = []
    for fname in sorted(os.listdir(dirpath)):
        if fname.endswith(".fjs"):
            full = os.path.join(dirpath, fname)
            if os.path.getsize(full) > 0:
                case_files.append(full)
    if not case_files:
        raise FileNotFoundError(f"No .fjs files found in {dirpath}")

    # 3) Make a local copy of env_paras and override batch_size
    val_paras = env_paras.copy()
    val_paras["batch_size"] = len(case_files)

    # 4) Create the env
    return gym.make(
        "fjsp-v0",
        case=case_files,
        env_paras=val_paras,
        data_source="file"
    )


def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state

    done = False
    dones = env.done_batch
    while not done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, makespans,emissions, dones = env.step(actions)

        done = dones.all()

    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    makespan_mean = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    
    # Emission metrics
    emission_batch = copy.deepcopy(env.total_emission_batch)
    emission_mean = copy.deepcopy(emission_batch.mean())
    
    env.reset()

    print('validating time: ', time.time() - start, '\n')
    
    # Return the 4 values train.py expects
    return makespan_mean, makespan_batch, emission_mean, emission_batch