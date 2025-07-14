import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model as PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env

from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    setup_seed(0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)

    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras.get("valid_batch_size", 100)

    print(f"[{time.strftime('%H:%M:%S')}] [Setup] Running numerical-only HGNN+PPO. LLM/Text fusion is DISABLED.")
    
    memories = PPO_model.Memory()
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    ppo_num_envs = env_paras.get("batch_size", 1) 
    model = PPO_model.PPO(model_paras, train_paras, num_envs=ppo_num_envs)

    env_valid = get_validate_env(env_valid_paras) 
    
    num_jobs_config = env_paras["num_jobs"]
    num_mas_config = env_paras["num_mas"]   
    opes_per_job_min_config = int(num_mas_config * 0.8) 
    opes_per_job_max_config = int(num_mas_config * 1.2) 
    
    best_models = deque()
    maxlen = train_paras.get("max_saved_models", 1) 
    composite_score_best = float('inf')
    
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])
        # Add windows for emission and composite score
        try:
            viz.line(X=np.array([0]), Y=np.array([0]), win='window_emission_valid_numeric', name='avg_emission_valid', opts=dict(title='Avg Emission of Valid (Numeric)'))
            viz.line(X=np.array([0]), Y=np.array([0]), win='window_composite_score_valid_numeric', name='avg_composite_score_valid', opts=dict(title='Avg Composite Score of Valid (Numeric)'))
            viz.line(X=np.array([0]), Y=np.array([0]), win='window_makespan_valid_numeric', name='avg_makespan_valid', opts=dict(title='Avg Makespan of Valid (Numeric)'))
        except Exception as e:
            print(f"Visdom warning (can be ignored if windows exist): {e}")

    str_time_main_run  = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path_train = f'./save/train_{str_time_main_run}_numerical_only'
    os.makedirs(save_path_train, exist_ok=True)
    
    # Setup Excel file paths for all three metrics
    file_ave_makespan_path = os.path.join(save_path_train, f'training_avg_makespan_{str_time_main_run}.xlsx')
    file_100_makespan_path = os.path.join(save_path_train, f'training_100_makespan_{str_time_main_run}.xlsx')
    file_ave_emission_path = os.path.join(save_path_train, f'training_avg_emission_{str_time_main_run}.xlsx')
    file_ave_composite_path = os.path.join(save_path_train, f'training_avg_composite_score_{str_time_main_run}.xlsx')

    # Setup lists to collect data for all three metrics
    collected_avg_makespans_for_excel = [] 
    collected_all_instance_makespans_tensors_for_excel  = [] 
    collected_avg_emissions_for_excel = []
    collected_avg_composite_scores_for_excel = []

    # Get weights for composite score calculation
    #alpha_makespan_weight = train_paras.get("alpha_makespan_weight", 0.5)
    beta_emission_weight = train_paras.get("lambda_penalty", 0.5)
    alpha_makespan_weight = 1 - beta_emission_weight  # Ensure weights sum to 1
    #print(f"[{time.strftime('%H:%M:%S')}] [Setup] Using lambda penalty for emission: {beta_emission_weight}")
    print(f"[{time.strftime('%H:%M:%S')}] [Setup] Composite score weights: Alpha (Makespan)={alpha_makespan_weight}, Beta (Emission)={beta_emission_weight}")

    main_start_time_training_loop = time.time()
    env_train = None 
    total_iters_train = train_paras["max_iterations"]
    parallel_iter_train = train_paras["parallel_iter"]

    for i_iter in tqdm(range(1, total_iters_train + 1), desc="🏋️ Training Progress (Numerical Only)", ncols=100):
        iter_start_time_current_loop = time.time()

        if (i_iter - 1) % parallel_iter_train == 0:
            current_nums_ope_for_env = [random.randint(opes_per_job_min_config, opes_per_job_max_config) for _ in range(num_jobs_config)]
            case_for_env = CaseGenerator(num_jobs_config, num_mas_config, opes_per_job_min_config, opes_per_job_max_config, nums_ope=current_nums_ope_for_env)
            if env_train and hasattr(env_train, 'close') and callable(env_train.close): env_train.close()
            env_train = gym.make('fjsp-v0', case=case_for_env, env_paras=env_paras)
        
        state_for_ppo_rollout_train = env_train.state 

        dones_batch_tracker_in_train = env_train.done_batch 
        while not dones_batch_tracker_in_train.all().item():
            with torch.no_grad():
                decoded_actions_tensor_train = model.policy_old.act(state_for_ppo_rollout_train, memories, dones_batch_tracker_in_train)

            if decoded_actions_tensor_train.nelement() > 0:
                next_state_obj_after_step_train, makespans_this_step_train, emissions_this_step_train, new_dones_batch_tracker_train = env_train.step(decoded_actions_tensor_train)
                
                memories.makespans.append(makespans_this_step_train)
                memories.emissions.append(emissions_this_step_train)
                memories.is_terminals.append(new_dones_batch_tracker_train)
                
                dones_batch_tracker_in_train = new_dones_batch_tracker_train
                state_for_ppo_rollout_train = next_state_obj_after_step_train 
            else: 
                print(f"[{time.strftime('%H:%M:%S')}] [Warning] Iter {i_iter}: Decoded actions tensor is empty during rollout. Breaking rollout.")
                break 
            if dones_batch_tracker_in_train.all().item(): break
        
        env_train.reset() 

        if i_iter % train_paras["update_timestep"] == 0:
            if memories.makespans:
                loss_train, ppo_reward_metric_train, _ = model.update(memories, env_paras, train_paras)
                if is_viz: 
                    viz.line(X=np.array([i_iter]), Y=np.array([ppo_reward_metric_train]), win='window0', update='append', opts=dict(title='reward of envs'))
                    viz.line(X=np.array([i_iter]), Y=np.array([loss_train]), win='window1', update='append', opts=dict(title='loss of envs'))
            else:
                print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Skipping PPO Update. No rewards in memory.")
            memories.clear_memory()

        if i_iter % train_paras["save_timestep"] == 0:
            print(f"\n[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Validating policy...")
            
            # Unpack the 4 return values from the corrected validate function
            vali_makespan_avg, vali_makespan_batch, vali_emission_avg, vali_emission_batch = validate(
                env_valid_paras, env_valid, model.policy_old
            )
            
            # Calculate composite score internally
            current_composite_score_val = alpha_makespan_weight * vali_makespan_avg.item() + beta_emission_weight * vali_emission_avg.item()

            # Collect all metrics for saving
            collected_avg_makespans_for_excel.append(vali_makespan_avg.item())
            collected_all_instance_makespans_tensors_for_excel.append(vali_makespan_batch.clone())
            collected_avg_emissions_for_excel.append(vali_emission_avg.item())
            collected_avg_composite_scores_for_excel.append(current_composite_score_val)

            print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: "
                  f"Validation Makespan: {vali_makespan_avg.item():.2f}, "
                  f"Emission: {vali_emission_avg.item():.2f}, "
                  f"Composite Score: {current_composite_score_val:.2f}")

            if current_composite_score_val < composite_score_best:
                composite_score_best = current_composite_score_val
                print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: New best composite score: {composite_score_best:.2f}")
                
                if len(best_models) >= maxlen and maxlen > 0 : 
                    delete_model_file_train = best_models.popleft() 
                    if os.path.exists(delete_model_file_train): os.remove(delete_model_file_train)
                
                save_filename_base_train = f"save_best_{num_jobs_config}_{num_mas_config}_{i_iter}_numerical_only" 
                model_save_file_train = os.path.join(save_path_train, f"{save_filename_base_train}.pt")
                
                best_models.append(model_save_file_train) 
                torch.save(model.policy.state_dict(), model_save_file_train)
                print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Saved new best model (numerical only) to {model_save_file_train}.")

            if is_viz:
                viz.line(X=np.array([i_iter]), Y=np.array([vali_makespan_avg.item()]), win='window_makespan_valid_numeric', update='append')
                viz.line(X=np.array([i_iter]), Y=np.array([vali_emission_avg.item()]), win='window_emission_valid_numeric', update='append')
                viz.line(X=np.array([i_iter]), Y=np.array([current_composite_score_val]), win='window_composite_score_valid_numeric', update='append')
        
        elapsed_total_main_train = time.time() - main_start_time_training_loop
        iter_time_taken_current_loop = time.time() - iter_start_time_current_loop 
        eta_main_train = (elapsed_total_main_train / i_iter) * (total_iters_train - i_iter) if i_iter > 0 else 0
        if i_iter % 100 == 0 or i_iter == total_iters_train:
            tqdm.write(f"⏱ Iter {i_iter}/{total_iters_train} | 🕒 IterTime: {iter_time_taken_current_loop:.2f}s | 🕘 Elapsed: {elapsed_total_main_train/60:.1f}m | ⏳ ETA: {eta_main_train/60:.1f}m")
    
    # After the loop, save all collected data to Excel files
    with pd.ExcelWriter(file_ave_makespan_path, engine='openpyxl') as writer_ave_makespan, \
         pd.ExcelWriter(file_100_makespan_path, engine='openpyxl') as writer_100_makespan, \
         pd.ExcelWriter(file_ave_emission_path, engine='openpyxl') as writer_ave_emission, \
         pd.ExcelWriter(file_ave_composite_path, engine='openpyxl') as writer_ave_composite:

        validation_iterations_excel_list = []
        if train_paras.get("save_timestep", 0) > 0 and train_paras.get("max_iterations", 0) >= train_paras.get("save_timestep", 0):
            validation_iterations_excel_list = list(range(train_paras["save_timestep"], 
                                                      train_paras["max_iterations"] + 1, 
                                                      train_paras["save_timestep"]))
        
        if collected_avg_makespans_for_excel and validation_iterations_excel_list:
            max_len = min(len(validation_iterations_excel_list), len(collected_avg_makespans_for_excel))
            pd.DataFrame({'iterations': validation_iterations_excel_list[:max_len], 
                        'avg_makespan': collected_avg_makespans_for_excel[:max_len]}
                        ).to_excel(writer_ave_makespan, sheet_name='Sheet1', index=False, header=True)

        if collected_avg_emissions_for_excel and validation_iterations_excel_list:
            max_len_em = min(len(validation_iterations_excel_list), len(collected_avg_emissions_for_excel))
            pd.DataFrame({'iterations': validation_iterations_excel_list[:max_len_em], 
                        'avg_emission': collected_avg_emissions_for_excel[:max_len_em]}
                        ).to_excel(writer_ave_emission, sheet_name='Sheet1', index=False, header=True)
        
        if collected_avg_composite_scores_for_excel and validation_iterations_excel_list:
            max_len_comp = min(len(validation_iterations_excel_list), len(collected_avg_composite_scores_for_excel))
            pd.DataFrame({'iterations': validation_iterations_excel_list[:max_len_comp], 
                        'avg_composite_score': collected_avg_composite_scores_for_excel[:max_len_comp]}
                        ).to_excel(writer_ave_composite, sheet_name='Sheet1', index=False, header=True)

        if collected_all_instance_makespans_tensors_for_excel and validation_iterations_excel_list:
            num_val_instances_excel_final = env_valid_paras["batch_size"] 
            columns_100_excel_final_names = [f"Instance_{k}" for k in range(num_val_instances_excel_final)]
            data_for_df_100_final = []
            max_len_100 = min(len(validation_iterations_excel_list), len(collected_all_instance_makespans_tensors_for_excel))
            iterations_for_df_100 = validation_iterations_excel_list[:max_len_100]
            for idx, tensor_run_final in enumerate(collected_all_instance_makespans_tensors_for_excel[:max_len_100]):
                iteration_num_final = iterations_for_df_100[idx]
                row_data_payload_final = tensor_run_final.cpu().numpy().flatten()[:num_val_instances_excel_final].tolist()
                while len(row_data_payload_final) < num_val_instances_excel_final: row_data_payload_final.append(np.nan)
                data_for_df_100_final.append([iteration_num_final] + row_data_payload_final)
            if data_for_df_100_final:
                df_100_columns_final_with_iter = ["iterations"] + columns_100_excel_final_names
                pd.DataFrame(data_for_df_100_final, columns=df_100_columns_final_with_iter).to_excel(writer_100_makespan, sheet_name='Sheet1', index=False, header=True)
  

    import shutil

    # 获取当前 config 中的 emission rates 和 lambda
    #env_paras = cfg["env_paras"]
    #train_paras = cfg["train_paras"]
    rate_list = env_paras.get("machine_emission_rates", [])
    lambda_val = train_paras.get("lambda_penalty", 0.0)

    # 构造路径标签：emission rates + lambda
    tag_str = "_".join(f"{r:.1f}" for r in rate_list)
    lambda_tag = str(lambda_val).replace(".", "_")
    model_dir = f"save/em_{tag_str}_lambda_{lambda_tag}"
    os.makedirs(model_dir, exist_ok=True)

    # 复制模型到指定路径
    dst_path = os.path.join(model_dir, "gnn.pt")  # ← 如果是 LLM 模型改成 llm.pt
    shutil.copy(model_save_file_train, dst_path)
    print(f"[✓] Copied best model to {dst_path}")


    total_training_time_main_final_run = time.time() - main_start_time_training_loop
    print(f"\n🏁 Training completed in {total_training_time_main_final_run:.1f} seconds ({total_training_time_main_final_run/60:.1f} min) using numerical features only.")

if __name__ == '__main__':
    main()