import copy
import json
import os
import random
import time as time_module 
import glob # Can be an alternative for listing model files

import gym
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import pynvml # Optional, for GPU monitoring
import PPO_model as PPO_model 
from env.load_data import nums_detec 
from tqdm import tqdm # Moved import to top for standard practice
import matplotlib.pyplot as plt # Moved import to top

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

def main(): 
    str_time = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime(time_module.time())) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    if device.type=='cuda': 
        torch.cuda.set_device(device) 
        torch.set_default_tensor_type('torch.cuda.FloatTensor') 
    else: 
        torch.set_default_tensor_type('torch.FloatTensor') 
    print("PyTorch device: ", device.type) 
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False) 

    with open("./config.json", 'r') as load_f: 
        load_dict = json.load(load_f) 
    env_paras = load_dict["env_paras"] 
    model_paras = load_dict["model_paras"] 
    train_paras = load_dict["train_paras"] 
    test_paras = load_dict["test_paras"] 
    env_paras["device"] = device 
    model_paras["device"] = device 
    env_test_paras = copy.deepcopy(env_paras) 
    num_ins = test_paras["num_ins"] 
    if test_paras["sample"]: 
        env_test_paras["batch_size"] = test_paras["num_sample"] 
    else: 
        env_test_paras["batch_size"] = 1 
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2 
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"] 

    data_path = "./data_test/{0}/".format(test_paras["data_path"]) 
    test_files = os.listdir(data_path) 
    test_files.sort(key=lambda x: x[:-4]) 
    test_files = test_files[:num_ins] 

    memories = PPO_model.Memory() 
    model = PPO_model.PPO(model_paras, train_paras) 
    rules = test_paras["rules"] 
    envs = []  

    if "DRL" in rules: 
        for root, ds, fs in os.walk('./model/'): 
            for f_name in fs: 
                if f_name.endswith('.pt'): 
                    rules.append(f_name) 
    if len(rules) > 1: # Check if "DRL" is still in rules after appending, and if it's not the only one
        try:
            rules.remove("DRL") # Remove "DRL" string placeholder if actual model files were added
        except ValueError:
            pass # "DRL" wasn't in the list (e.g. if it was already only model names)


    save_path = './save/test_{0}'.format(str_time) 
    os.makedirs(save_path, exist_ok=True) 
 
    makespan_file_path = os.path.join(save_path, f"makespan_{str_time}.xlsx") 
    time_file_path = os.path.join(save_path, f"time_{str_time}.xlsx") 
    file_name_column = [test_files[i] for i in range(num_ins)] 
    data_file_df = pd.DataFrame(file_name_column, columns=["file_name"]) 
    
    with pd.ExcelWriter(makespan_file_path) as writer_makespan_init, pd.ExcelWriter(time_file_path) as writer_time_init: 
        data_file_df.to_excel(writer_makespan_init, sheet_name='Sheet1', index=False) 
        data_file_df.to_excel(writer_time_init, sheet_name='Sheet1', index=False) 
    
    # Optional: Prepare emission file
    emission_file_path = os.path.join(save_path, f"emission_{str_time}.xlsx")
    with pd.ExcelWriter(emission_file_path) as writer_emission_init: # Create the file with instance names
       data_file_df.to_excel(writer_emission_init, sheet_name='Sheet1', index=False)

    start_run_time = time_module.time() 
    total_steps = len(rules) * num_ins 
    step_count = 0 

    # Use a single ExcelWriter context for appending data to avoid issues with multiple openings
    with pd.ExcelWriter(makespan_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer_makespan, \
         pd.ExcelWriter(time_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer_time, \
         pd.ExcelWriter(emission_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer_emission:

        for i_rules in tqdm(range(len(rules)), desc="📊 Testing Rules"): 
            rule = rules[i_rules] 

            if rule.endswith('.pt'): 
                model_path_full = os.path.join('model', rule) 
                print(f"\n📥 Loading checkpoint {model_path_full}") 
                if device.type == 'cuda': 
                    model_CKPT = torch.load(model_path_full) 
                else: 
                    model_CKPT = torch.load(model_path_full, map_location='cpu') 

                model.policy.load_state_dict(model_CKPT) 
                model.policy_old.load_state_dict(model_CKPT) 
                print(f"\n📥 Loaded model checkpoint: {rule}") 

            print(f"🧪 Testing rule: {rule}") 
            step_time_last = time_module.time() 
            makespans_list, times_list, emissions_list = [], [], [] # Added emissions_list

            for i_ins in tqdm(range(num_ins), desc=f"→ Rule '{rule}' progress", leave=False): 
                inst_start_time = time_module.time() 
                test_file_path_full = os.path.join(data_path, test_files[i_ins])

                with open(test_file_path_full) as file_object: 
                    line = file_object.readlines() 
                    ins_num_jobs, ins_num_mas, _ = nums_detec(line) 
                current_env_test_paras = copy.deepcopy(env_test_paras) # Use a copy to modify per instance
                current_env_test_paras["num_jobs"] = ins_num_jobs 
                current_env_test_paras["num_mas"] = ins_num_mas 
                
                # Simplified env creation: always create a new one for this test instance
                case_list = [test_file_path_full] * current_env_test_paras["batch_size"] # Handles sample or single
                env = gym.make('fjsp-v0', case=case_list, env_paras=current_env_test_paras, data_source='file') 
                # print(f"🌱 Created/Reused env for {test_files[i_ins]}") # Removed envs list complexity

                if test_paras["sample"]: 
                    makespan_val, total_emissions_val, time_re = schedule(env, model, memories, env_name=test_files[i_ins][:-4], flag_sample=True, str_time=str_time)
                    makespans_list.append(torch.min(makespan_val.cpu())) 
                    emissions_list.append(torch.min(total_emissions_val.cpu())) 
                    times_list.append(time_re) 
                else: 
                    time_s_avg, makespan_s_avg, emission_s_avg = [], [], []
                    for j in range(test_paras["num_average"]): 
                        makespan_val, total_emissions_val, time_re = schedule(env, model, memories, env_name=test_files[i_ins][:-4] + f"_avg{j}", str_time=str_time) 
                        makespan_s_avg.append(makespan_val.cpu()) 
                        emission_s_avg.append(total_emissions_val.cpu())
                        time_s_avg.append(time_re) 
                        env.reset() 
                    makespans_list.append(torch.mean(torch.stack(makespan_s_avg) if makespan_s_avg else torch.tensor(float('nan')))) 
                    emissions_list.append(torch.mean(torch.stack(emission_s_avg) if emission_s_avg else torch.tensor(float('nan'))))
                    times_list.append(torch.mean(torch.tensor(time_s_avg, dtype=torch.float) if time_s_avg else torch.tensor(float('nan'))))

                step_count += 1 
                inst_end_time = time_module.time() 
                elapsed = time_module.time() - start_run_time 
                eta = (elapsed / step_count) * (total_steps - step_count) if step_count > 0 else 0 
                print(f"✅ Finished env {i_ins} ({test_files[i_ins]}) | ⏱ {inst_end_time - inst_start_time:.2f}s | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m") 
                
                env.close() # Close env if newly created each time

            print(f"✔️ Finished rule '{rule}' in {time_module.time() - step_time_last:.2f}s") 

            # Save results for the current rule
            df_makespan_results = pd.DataFrame([m.tolist() if isinstance(m, torch.Tensor) else m for m in makespans_list], columns=[rule])
            df_time_results = pd.DataFrame([t if not isinstance(t, torch.Tensor) else t.tolist() for t in times_list], columns=[rule])
            df_emission_results = pd.DataFrame([e.tolist() if isinstance(e, torch.Tensor) else e for e in emissions_list], columns=[rule])

            start_col_idx = writer_makespan.sheets['Sheet1'].max_column
            df_makespan_results.to_excel(writer_makespan, sheet_name='Sheet1', index=False, header=True, startcol=start_col_idx)
            
            start_col_idx = writer_time.sheets['Sheet1'].max_column
            df_time_results.to_excel(writer_time, sheet_name='Sheet1', index=False, header=True, startcol=start_col_idx)
            
            start_col_idx = writer_emission.sheets['Sheet1'].max_column # Assuming emission file was prepped with instance names
            df_emission_results.to_excel(writer_emission, sheet_name='Sheet1', index=False, header=True, startcol=start_col_idx)


    total_time_run = time_module.time() - start_run_time 
    print(f"\n🏁 All testing done in {total_time_run:.1f} seconds ({total_time_run/60:.1f} minutes)") 


def schedule(env, model, memories, env_name='tbc', flag_sample=False, str_time=None): 
    state = env.state 
    dones = env.done_batch 
    done = dones.all() 
    last_time = time_module.time() 
    i_step = 0 
    
    memories.clear_memory()

    while not done: 
        i_step += 1 
        with torch.no_grad(): 
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False) 
        
        action_device = env.device if hasattr(env, 'device') and env.device is not None else 'cpu'
        state, makespans,emissions, dones = env.step(actions.to(action_device)) 
        done = dones.all() 

    spend_time = time_module.time() - last_time 
    print(f"Instance {env_name} scheduled in {spend_time:.2f}s with {i_step} steps.") 
    # print("Number of jobs in the environment:", env.num_jobs) # Less verbose

    is_valid, schedule_batch_tensor = env.validate_gantt()
    
    final_makespan = copy.deepcopy(env.makespan_batch)
    print(f"Final makespan for {env_name}: {final_makespan}") # Print final makespan for clarity
    final_total_emissions = copy.deepcopy(env.total_emission_batch) # Capture emissions
    print(f"Final total emissions for {env_name}: {final_total_emissions}") # Print final emissions for clarity
    if not is_valid: 
        print(f"Scheduling Error for {env_name}！！！！！！") 

    if is_valid: 
        df = parse_schedule(schedule_batch_tensor, env) 
        
        if final_makespan.numel() == 1:
            makespan_for_plot = final_makespan.item()
        elif final_makespan.numel() > 0 : 
             makespan_for_plot = final_makespan.mean().item() 
        else: 
            makespan_for_plot = float('nan') # Use NaN for invalid makespan
            # print(f"Warning: final_makespan for {env_name} is empty or invalid for plotting.")

        if not df.empty: 
            plot_gantt(df, env_name=env_name, makespan=makespan_for_plot, str_time=str_time) 
            save_report(df, env_name=env_name, makespan=makespan_for_plot, str_time=str_time) 
        else:
            print(f"DataFrame for Gantt is empty for {env_name}. Skipping plot and report.")
    else: 
        print(f"Invalid schedule for {env_name}! Cannot generate Gantt chart.") 

    return final_makespan, final_total_emissions, spend_time # Return emissions


def parse_schedule(schedule_batch_tensor, env): 
    schedule_data = [] 
    actual_batch_size_in_tensor = schedule_batch_tensor.shape[0]

    for b_idx_in_tensor in range(actual_batch_size_in_tensor):
        if b_idx_in_tensor >= env.batch_size: 
            # print(f"Warning: b_idx_in_tensor {b_idx_in_tensor} is out of bounds for env attributes. Skipping this batch item in parse_schedule.")
            continue
            
        # Ensure env.nums_opes has items for b_idx_in_tensor
        if b_idx_in_tensor >= len(env.nums_opes):
            # print(f"Warning: b_idx_in_tensor {b_idx_in_tensor} out of range for env.nums_opes. Skipping.")
            continue
        num_actual_ops_instance = env.nums_opes[b_idx_in_tensor].item()

        for op_idx_global in range(num_actual_ops_instance):
            # Ensure op_idx_global is within bounds for schedule_batch_tensor's second dim
            if op_idx_global >= schedule_batch_tensor.shape[1]:
                # print(f"Warning: op_idx_global {op_idx_global} out of bounds for schedule_batch_tensor. Skipping.")
                continue
            step = schedule_batch_tensor[b_idx_in_tensor, op_idx_global, :]
            
            if step[0].item() == 1 and step[2].item() < step[3].item(): 
                status_val = step[0].item()
                machine_id_val = step[1].item()
                start_time_val = step[2].item()
                end_time_val = step[3].item()
                
                if op_idx_global >= env.opes_appertain_batch.shape[1]:
                    # print(f"Warning: op_idx_global {op_idx_global} out of bounds for env.opes_appertain_batch. Assigning default Job ID.")
                    job_id_val = -1 # Default or placeholder Job ID
                else:
                    job_id_val = env.opes_appertain_batch[b_idx_in_tensor, op_idx_global].item()

                schedule_data.append([
                    status_val, machine_id_val, start_time_val, end_time_val, job_id_val 
                ])

    if not schedule_data: 
        return pd.DataFrame(columns=["Status", "Machine ID", "Start Time", "End Time", "Job ID"])

    cols = ["Status", "Machine ID", "Start Time", "End Time", "Job ID"] 
    df = pd.DataFrame(schedule_data, columns=cols) 
    return df 


def plot_gantt(df, env_name, makespan, str_time): 
    if df.empty or not all(col in df.columns for col in ["Machine ID", "Start Time", "End Time", "Job ID"]):
        # print(f"DataFrame is empty or missing required columns for Gantt chart: {env_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 7)) 
    df_filtered = df[df["Job ID"] >= 0].copy()
    if df_filtered.empty :
        # print(f"No valid jobs with non-negative Job IDs to plot for {env_name}")
        plt.close(fig)
        return

    df_filtered["Job ID"] = df_filtered["Job ID"].astype(int)
    df_filtered["Machine ID"] = df_filtered["Machine ID"].astype(int)
    unique_job_ids = sorted(df_filtered["Job ID"].unique())
    
    if unique_job_ids:
        cmap = plt.cm.get_cmap('tab20', len(unique_job_ids))
        colors_dict = {job_id: cmap(i) for i, job_id in enumerate(unique_job_ids)}
    else:
        colors_dict = {}

    machine_ids_numeric = sorted(df_filtered["Machine ID"].unique()) 

    for _, row in df_filtered.iterrows(): 
        job_id_val = row["Job ID"]
        # Ensure start_time and end_time are valid numbers
        if pd.isna(row["Start Time"]) or pd.isna(row["End Time"]) or row["End Time"] <= row["Start Time"]:
            continue # Skip invalid duration bars
            
        ax.barh( 
            y=row["Machine ID"], 
            width=row["End Time"] - row["Start Time"], 
            left=row["Start Time"], 
            color=colors_dict.get(job_id_val, 'gray'), 
            edgecolor="black", 
            label=f"Job {job_id_val}" if job_id_val not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else ""
        )

    title = f"Gantt Chart: {env_name}, Makespan={makespan:.2f}" if not pd.isna(makespan) else f"Gantt Chart: {env_name}"
    ax.set_title(title) 
    ax.set_xlabel("Time") 
    ax.set_ylabel("Machine ID") 
    
    if machine_ids_numeric: 
        ax.set_yticks(machine_ids_numeric) 
        ax.set_yticklabels([f"Machine {mid}" for mid in machine_ids_numeric]) 
    
    handles, labels = ax.get_legend_handles_labels()
    if handles: 
      ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=min(6, len(unique_job_ids) if unique_job_ids else 1), fancybox=True, shadow=True) 

    plt.grid(axis="x", linestyle="--", alpha=0.7) 
    fig.subplots_adjust(bottom=0.25 if handles else 0.1) # Adjust bottom margin for legend

    save_dir = os.path.join('.', 'save', f'test_{str_time}', 'ganttresult') 
    os.makedirs(save_dir, exist_ok=True) 
    sanitized_env_name = env_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    makespan_str = f"{makespan:.2f}" if not pd.isna(makespan) else "N_A"
    save_file_path = os.path.join(save_dir, f'{sanitized_env_name}_makespan_{makespan_str}.png') 

    plt.savefig(save_file_path, dpi=300) 
    #print(f"Gantt chart saved as {save_file_path}") 
    plt.close(fig) 


def save_report(df, env_name, makespan, str_time): 
    if df.empty:
        # print(f"DataFrame is empty. No report to save for {env_name}.")
        return

    save_dir = os.path.join('.', 'save', f'test_{str_time}', 'report') 
    os.makedirs(save_dir, exist_ok=True) 
    sanitized_env_name = env_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    makespan_str = f"{makespan:.2f}" if not pd.isna(makespan) else "N_A"
    save_file_path = os.path.join(save_dir, f'{sanitized_env_name}_makespan_{makespan_str}.json') 

    report_data = { 
        "env_name": env_name, 
        "makespan": makespan if not pd.isna(makespan) else None, 
        "schedule": df.to_dict(orient='records') 
    }
    with open(save_file_path, 'w') as f_json: 
        json.dump(report_data, f_json, indent=4) 
    #print(f"Report saved as {save_file_path}") 

if __name__ == '__main__': 
    setup_seed(42) # Example seed
    main()