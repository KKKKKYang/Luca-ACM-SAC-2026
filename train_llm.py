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

import text_features
import torch.nn as nn
from fusion import GatedOpFusion

from tqdm import tqdm

# --- BEGIN: Verbosity Control & Globals for Dynamic Prompting ---
verbose_dynamic_prompting = True
dynamic_prompt_feedback_log = []
op_extra_prompts = {}
# --- END ---

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# --- BEGIN: New Detailed Dynamic Prompting Functions ---
def update_prompts_with_detailed_feedback(feedback_log, 
                                          makespan_risk_threshold=-0.2, 
                                          high_emission_threshold=15.0):
    """
    Generates specific prompts based on separate makespan and emission feedback.
    """
    global op_extra_prompts, verbose_dynamic_prompting
    if verbose_dynamic_prompting:
        print(f"[{time.strftime('%H:%M:%S')}] [DP Detailed] Running update_prompts_with_detailed_feedback. Log length: {len(feedback_log)}")

    op_feedback_accumulator = {}
    for entry in feedback_log:
        op_id = entry['op_id']
        if op_id not in op_feedback_accumulator:
            op_feedback_accumulator[op_id] = {'makespan_impacts': [], 'emission_impacts': []}
        
        op_feedback_accumulator[op_id]['makespan_impacts'].append(entry['makespan_impact'])
        op_feedback_accumulator[op_id]['emission_impacts'].append(entry['emission_impact'])

    ops_feedback_changed_count = 0
    
    for op_id, data in op_feedback_accumulator.items():
        if not data['makespan_impacts']: continue
        
        avg_makespan_impact = sum(data['makespan_impacts']) / len(data['makespan_impacts'])
        avg_emission_impact = sum(data['emission_impacts']) / len(data['emission_impacts'])
        #print('len(data[\'makespan_impacts\']):', len(data['makespan_impacts']))
        feedback_parts = []
        # Analyze Makespan Impact (makespan_impact is `old - new`, so a negative value is bad)
        if avg_makespan_impact < makespan_risk_threshold:
            feedback_parts.append("Hint:HighMakespanRisk")

        # Analyze Emission Impact (raw emission, higher is worse)
        if avg_emission_impact > high_emission_threshold:
            feedback_parts.append("Hint:HighEmission")
        
        new_feedback_str = ""
        if feedback_parts:
            new_feedback_str = "; " + "; ".join(feedback_parts)
        
        existing_feedback = op_extra_prompts.get(op_id, None)

        if new_feedback_str != existing_feedback:
            if new_feedback_str:
                op_extra_prompts[op_id] = new_feedback_str
                if verbose_dynamic_prompting: print(f"[{time.strftime('%H:%M:%S')}] [DP Detailed] Op_id {op_id}: SET/UPDATED prompt to '{new_feedback_str}'")
            elif existing_feedback is not None:
                del op_extra_prompts[op_id]
                if verbose_dynamic_prompting: print(f"[{time.strftime('%H:%M:%S')}] [DP Detailed] Op_id {op_id}: CLEARED prompt.")
            ops_feedback_changed_count += 1
    
    if verbose_dynamic_prompting:
        print(f"[{time.strftime('%H:%M:%S')}] [DP Detailed] Update finished. Feedback changed for {ops_feedback_changed_count} ops.")

def build_dynamic_operation_descriptions(state):
    global op_extra_prompts, verbose_dynamic_prompting
    if verbose_dynamic_prompting:
        num_potential_ops_in_state = state.feat_opes_batch.shape[0] * state.feat_opes_batch.shape[2]
        active_warnings = sum(1 for op_id_key_str in op_extra_prompts
                              if (isinstance(op_id_key_str, int) and 0 <= op_id_key_str < num_potential_ops_in_state) or \
                                 (isinstance(op_id_key_str, str) and op_id_key_str.isdigit() and 0 <= int(op_id_key_str) < num_potential_ops_in_state)
                             )
        if len(op_extra_prompts) > 0 or active_warnings > 0:
            print(f"[{time.strftime('%H:%M:%S')}] [DP] Running build_dynamic_operation_descriptions. op_extra_prompts has {len(op_extra_prompts)} entries. Approx {active_warnings} potentially relevant.")

    descs = []
    B, _, num_ops = state.feat_opes_batch.shape

    est_starts = state.feat_opes_batch[:, 5, :].tolist()
    mean_times = state.feat_opes_batch[:, 2, :].tolist()
    rem_ops    = state.feat_opes_batch[:, 3, :].tolist()
    proc_times = state.proc_times_batch.tolist()
    eligible   = state.ope_ma_adj_batch.tolist()
    job_ids    = state.opes_appertain_batch.tolist()

    warnings_injected_count = 0
    for b_idx in range(B):
        for op_idx in range(num_ops):
            job    = job_ids[b_idx][op_idx]
            left   = int(rem_ops[b_idx][op_idx])
            est    = est_starts[b_idx][op_idx]
            dur    = mean_times[b_idx][op_idx]

            mts = [f"{m}:{proc_times[b_idx][op_idx][m]:.1f}" for m, ok in enumerate(eligible[b_idx][op_idx]) if ok]
            machines_str = "|".join(mts) if mts else "none"

            base_desc = (f"Job {job} ▸ Op {op_idx} — {left} ops left; est_start={est:.1f}, dur={dur:.1f}; machines={machines_str}")
            dynamic_warning = op_extra_prompts.get(op_idx, op_extra_prompts.get(str(op_idx), ""))

            if dynamic_warning:
                warnings_injected_count +=1
            descs.append(base_desc + ((" " + dynamic_warning) if dynamic_warning else ""))
    
    if verbose_dynamic_prompting and (warnings_injected_count > 0 or (B > 0 and num_ops > 0)):
        print(f"[{time.strftime('%H:%M:%S')}] [DP] build_dynamic_operation_descriptions finished. Total descriptions: {len(descs)}. Injected {warnings_injected_count} warnings.")
    return descs
# --- END: Dynamic Prompting Functions ---

def main():
    global verbose_dynamic_prompting, op_extra_prompts, dynamic_prompt_feedback_log
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

    fusion_model = GatedOpFusion(D_op=env_paras["ope_feat_dim"], D_text=text_features.get_sentence_embedding_dimension()).to(device)
    print(f"[{time.strftime('%H:%M:%S')}] [Setup] LLM/Fusion Method: GatedOpFusion model initialized.")
    
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
        try:
            viz.line(X=np.array([0]), Y=np.array([0]), win='window_emission_valid', name='avg_emission_valid', opts=dict(title='Avg Emission of Valid'))
            viz.line(X=np.array([0]), Y=np.array([0]), win='window_composite_score_valid', name='avg_composite_score_valid', opts=dict(title='Avg Composite Score of Valid'))
        except Exception as e:
            print(f"Visdom warning (can be ignored if windows exist): {e}")

    str_time_main_run  = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path_train = f'./save/train_{str_time_main_run}_llm_fusion_carbon'
    os.makedirs(save_path_train, exist_ok=True)
    
    file_ave_makespan_path = os.path.join(save_path_train, f'training_avg_makespan_{str_time_main_run}.xlsx')
    file_100_makespan_path = os.path.join(save_path_train, f'training_100_makespan_{str_time_main_run}.xlsx')
    file_ave_emission_path = os.path.join(save_path_train, f'training_avg_emission_{str_time_main_run}.xlsx')
    file_ave_composite_path = os.path.join(save_path_train, f'training_avg_composite_score_{str_time_main_run}.xlsx')

    collected_avg_makespans_for_excel = []
    collected_all_instance_makespans_tensors_for_excel  = []
    collected_avg_emissions_for_excel = []
    collected_avg_composite_scores_for_excel = []
    
    beta_emission_weight = train_paras.get("lambda_penalty", 0.5)
    alpha_makespan_weight = 1- beta_emission_weight

    print(f"[{time.strftime('%H:%M:%S')}] [Setup] Composite score weights: Alpha (Makespan)={alpha_makespan_weight}, Beta (Emission)={beta_emission_weight}")

    if verbose_dynamic_prompting:
        print(f"[{time.strftime('%H:%M:%S')}] [DP] Dynamic Prompting (Enhanced Thresholds) Verbose Logging ENABLED.")

    with pd.ExcelWriter(file_ave_makespan_path, engine='openpyxl') as writer_ave_makespan, \
         pd.ExcelWriter(file_100_makespan_path, engine='openpyxl') as writer_100_makespan, \
         pd.ExcelWriter(file_ave_emission_path, engine='openpyxl') as writer_ave_emission, \
         pd.ExcelWriter(file_ave_composite_path, engine='openpyxl') as writer_ave_composite:
        
        validation_iterations_excel_list = []
        save_interval = train_paras.get("save_timestep", 0)
        max_iters = train_paras.get("max_iterations", 0)

        if save_interval > 0 and max_iters >= save_interval:
            validation_iterations_excel_list = list(range(save_interval, max_iters + 1, save_interval))
        
        if validation_iterations_excel_list:
            header_df = pd.DataFrame({"iterations": validation_iterations_excel_list})
            header_df.to_excel(writer_ave_makespan, sheet_name='Sheet1', index=False, startrow=0, startcol=0)
            writer_ave_makespan.sheets['Sheet1'].cell(row=1, column=2).value = "avg_makespan"
            header_df.to_excel(writer_100_makespan, sheet_name='Sheet1', index=False, startrow=0, startcol=0)
            header_df.to_excel(writer_ave_emission, sheet_name='Sheet1', index=False, startrow=0, startcol=0)
            writer_ave_emission.sheets['Sheet1'].cell(row=1, column=2).value = "avg_emission"
            header_df.to_excel(writer_ave_composite, sheet_name='Sheet1', index=False, startrow=0, startcol=0)
            writer_ave_composite.sheets['Sheet1'].cell(row=1, column=2).value = "avg_composite_score"

    main_start_time_training_loop = time.time()
    env_train = None
    total_iters_train = train_paras["max_iterations"]
    parallel_iter_train = train_paras["parallel_iter"]
    dynamic_prompt_update_freq_train = train_paras.get("dynamic_prompt_update_freq", 20)

    machine_emission_rates_tensor = torch.tensor(env_paras.get("machine_emission_rates", []), device=device)

    for i_iter in tqdm(range(1, total_iters_train + 1), desc="🏋️ Training Progress (LLM+Fusion+Carbon)", ncols=100):
        iter_start_time_current_loop = time.time()

        if (i_iter - 1) % parallel_iter_train == 0:
            current_nums_ope_for_env = [random.randint(opes_per_job_min_config, opes_per_job_max_config) for _ in range(num_jobs_config)]
            case_for_env = CaseGenerator(num_jobs_config, num_mas_config, opes_per_job_min_config, opes_per_job_max_config, nums_ope=current_nums_ope_for_env)
            if env_train and hasattr(env_train, 'close') and callable(env_train.close): env_train.close()
            env_train = gym.make('fjsp-v0', case=case_for_env, env_paras=env_paras)
            if verbose_dynamic_prompting: print(f"\n[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Rebuilt training env. Resetting DP logs.")
            op_extra_prompts.clear()
            dynamic_prompt_feedback_log.clear()

        current_env_state_obj_for_train = env_train.state
        
        feat_opes_batch_numeric_for_train = current_env_state_obj_for_train.feat_opes_batch.clone()
        B_train_loop, _, num_ops_in_instance_loop_train = feat_opes_batch_numeric_for_train.shape
        
        all_descs_for_train = build_dynamic_operation_descriptions(current_env_state_obj_for_train)
        all_embs_dynamic_for_train = text_features.encode_op_descriptions(all_descs_for_train).to(device)
        text_embeddings_for_fusion_in_train = all_embs_dynamic_for_train.view(B_train_loop, num_ops_in_instance_loop_train, -1).permute(0, 2, 1)
        fused_feat_opes_batch_for_train = fusion_model(feat_opes_batch_numeric_for_train, text_embeddings_for_fusion_in_train)
        
        state_for_ppo_rollout_train = copy.deepcopy(current_env_state_obj_for_train)
        state_for_ppo_rollout_train.feat_opes_batch = fused_feat_opes_batch_for_train

        dones_batch_tracker_in_train = env_train.done_batch
        while not dones_batch_tracker_in_train.all().item():
            with torch.no_grad():
                decoded_actions_tensor_train = model.policy_old.act(state_for_ppo_rollout_train, memories, dones_batch_tracker_in_train)

            if decoded_actions_tensor_train.nelement() > 0:
                # --- START MODIFICATIONS for DETAILED PROMPTING ---
                # 1. CAPTURE STATE BEFORE THE STEP
                old_makespan_batch = env_train.makespan_batch.clone()
                active_indices_before_step = state_for_ppo_rollout_train.batch_idxes.clone()
                active_actions_opes = decoded_actions_tensor_train[0, :]
                active_actions_mas = decoded_actions_tensor_train[1, :]

                # 2. PERFORM THE STEP (env.step now returns 4 values)
                next_state_obj_after_step_train, makespans_this_step, emissions_this_step, new_dones_batch_tracker_train = env_train.step(decoded_actions_tensor_train)
                
                # 3. RE-CALCULATE COMPONENTS & LOG DETAILED FEEDBACK
                new_makespan_batch = env_train.makespan_batch
                
                proc_times_for_actions = env_train.proc_times_batch[active_indices_before_step, active_actions_opes.long(), active_actions_mas.long()]
                emission_rates_for_actions = machine_emission_rates_tensor[active_actions_mas.long()]
                emission_impact_batch = proc_times_for_actions * emission_rates_for_actions

                for k_env_idx in range(active_actions_opes.size(0)):
                    op_id_scheduled_train = active_actions_opes[k_env_idx].item()
                    original_batch_idx = active_indices_before_step[k_env_idx].item()
                    makespan_impact_val = (old_makespan_batch[original_batch_idx] - new_makespan_batch[original_batch_idx]).item()
                    emission_impact_val = emission_impact_batch[k_env_idx].item()
                    
                    dynamic_prompt_feedback_log.append({
                        'op_id': op_id_scheduled_train,
                        'makespan_impact': makespan_impact_val,
                        'emission_impact': emission_impact_val
                    })
                    #print('k_env_idx:', k_env_idx,'len(dynamic_prompt_feedback_log):', len(dynamic_prompt_feedback_log))
                    #print(f"[{time.strftime('%H:%M:%S')}] [DP Detailed] Op {op_id_scheduled_train}: Makespan Impact: {makespan_impact_val:.2f}, Emission Impact: {emission_impact_val:.2f}")
                    #print(f'dynamic_prompt_feedback_log[-1]: {dynamic_prompt_feedback_log[-1]}')  # Print the last entry for immediate feedback
                    #print('len(dynamic_prompt_feedback_log):', len(dynamic_prompt_feedback_log))
                # --- END MODIFICATIONS ---
                
                # Store separate reward components in memory
                memories.makespans.append(makespans_this_step)
                memories.emissions.append(emissions_this_step)
                memories.is_terminals.append(new_dones_batch_tracker_train)
                
                dones_batch_tracker_in_train = new_dones_batch_tracker_train
                state_for_ppo_rollout_train = next_state_obj_after_step_train
            else:
                if verbose_dynamic_prompting: print(f"[{time.strftime('%H:%M:%S')}] [Warning] Iter {i_iter}: Decoded actions tensor is empty. Breaking rollout.")
                break
            if dones_batch_tracker_in_train.all().item(): break
        
        env_train.reset()

        if i_iter % dynamic_prompt_update_freq_train == 0 and dynamic_prompt_feedback_log:
            if verbose_dynamic_prompting: print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Updating op_extra_prompts with DETAILED feedback.")
            update_prompts_with_detailed_feedback(dynamic_prompt_feedback_log)
            dynamic_prompt_feedback_log.clear()

        if i_iter % train_paras["update_timestep"] == 0:
            if memories.logprobs: # Check if there is anything to update
                # Unpack 3 values, discarding the last one
                loss_train, ppo_reward_metric_train, _ = model.update(memories, env_paras, train_paras)
                if verbose_dynamic_prompting: print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: PPO Update. Reward: {ppo_reward_metric_train:.3f}, Loss: {loss_train:.3f}")
                if is_viz:
                    viz.line(X=np.array([i_iter]), Y=np.array([ppo_reward_metric_train]), win='window0', update='append', opts=dict(title='reward of envs'))
                    viz.line(X=np.array([i_iter]), Y=np.array([loss_train]), win='window1', update='append', opts=dict(title='loss of envs'))
            else:
                if verbose_dynamic_prompting: print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Skipping PPO Update. No memories to process.")
            memories.clear_memory()

        if i_iter % train_paras["save_timestep"] == 0:
            if verbose_dynamic_prompting: print(f"\n[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Validating policy...")
            
            # Unpack 4 values from the corrected validate function
            vali_makespan_avg, vali_makespan_batch, vali_emission_avg, vali_emission_batch = validate(env_valid_paras, env_valid, model.policy_old)
            
            current_composite_score_val = alpha_makespan_weight * vali_makespan_avg.item() + beta_emission_weight * vali_emission_avg.item()

            collected_avg_makespans_for_excel.append(vali_makespan_avg.item())
            collected_all_instance_makespans_tensors_for_excel.append(vali_makespan_batch.clone())
            collected_avg_emissions_for_excel.append(vali_emission_avg.item())
            collected_avg_composite_scores_for_excel.append(current_composite_score_val)

            if verbose_dynamic_prompting:
                print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Validation Avg Makespan: {vali_makespan_avg.item():.2f}, Avg Emission: {vali_emission_avg.item():.2f}, Avg Composite Score: {current_composite_score_val:.2f}")

            if current_composite_score_val < composite_score_best:
                composite_score_best = current_composite_score_val
                if verbose_dynamic_prompting:
                    print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: New best composite score: {composite_score_best:.2f} (Makespan: {vali_makespan_avg.item():.2f}, Emission: {vali_emission_avg.item():.2f})")
                
                if len(best_models) >= maxlen and maxlen > 0:
                    delete_model_file_train = best_models.popleft()
                    if os.path.exists(delete_model_file_train): os.remove(delete_model_file_train)
                
                save_filename_base_train = f"save_best_score_{num_jobs_config}_{num_mas_config}_{i_iter}_llm_fusion_carbon"
                model_save_file_train = os.path.join(save_path_train, f"{save_filename_base_train}.pt")
                
                best_models.append(model_save_file_train)
                torch.save(model.policy.state_dict(), model_save_file_train)
                
                prompts_save_file_train = os.path.join(save_path_train, f"{save_filename_base_train}_op_extra_prompts_detailed.json")
                prompts_to_save_train = {str(k): v for k, v in op_extra_prompts.items()}
                with open(prompts_save_file_train, 'w') as f_prompts_train:
                    json.dump(prompts_to_save_train, f_prompts_train, indent=4)
                if verbose_dynamic_prompting: print(f"[{time.strftime('%H:%M:%S')}] [MainLoop] Iter {i_iter}: Saved new best model (by composite score) to {model_save_file_train} and DETAILED prompts.")

            if is_viz:
                viz.line(X=np.array([i_iter]), Y=np.array([vali_makespan_avg.item()]), win='window2', update='append', opts=dict(title='Makespan of Valid'))
                viz.line(X=np.array([i_iter]), Y=np.array([vali_emission_avg.item()]), win='window_emission_valid', name='avg_emission_valid', update='append', opts=dict(title='Avg Emission of Valid'))
                viz.line(X=np.array([i_iter]), Y=np.array([current_composite_score_val]), win='window_composite_score_valid', name='avg_composite_score_valid', update='append', opts=dict(title='Avg Composite Score of Valid'))
        
        elapsed_total_main_train = time.time() - main_start_time_training_loop
        iter_time_taken_current_loop = time.time() - iter_start_time_current_loop
        eta_main_train = (elapsed_total_main_train / i_iter) * (total_iters_train - i_iter) if i_iter > 0 else 0
        if i_iter % 10 == 0 or i_iter == total_iters_train:
            tqdm.write(f"⏱ Iter {i_iter}/{total_iters_train} | 🕒 IterTime: {iter_time_taken_current_loop:.2f}s | 🕘 Elapsed: {elapsed_total_main_train/60:.1f}m | ⏳ ETA: {eta_main_train/60:.1f}m")

    with pd.ExcelWriter(file_ave_makespan_path, engine='openpyxl') as writer_ave_makespan, \
         pd.ExcelWriter(file_100_makespan_path, engine='openpyxl') as writer_100_makespan, \
         pd.ExcelWriter(file_ave_emission_path, engine='openpyxl') as writer_ave_emission, \
         pd.ExcelWriter(file_ave_composite_path, engine='openpyxl') as writer_ave_composite:

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

    # import shutil

    # # 构造保存路径
    # tag_str = "_".join(f"{r:.2f}" for r in env_paras.get("machine_emission_rates", []))
    # src_path = model_save_file_train  # 你已经保存的模型路径
    # dst_path = f"save/llm_em_{tag_str}.pt"
    # shutil.copy(src_path, dst_path)
    # print(f"[✓] Copied best model to {dst_path}")
    
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
    dst_path = os.path.join(model_dir, "llm.pt")  # ← 如果是 LLM 模型改成 llm.pt
    shutil.copy(model_save_file_train, dst_path)
    print(f"[✓] Copied best model to {dst_path}")

    total_training_time_main_final_run = time.time() - main_start_time_training_loop
    print(f"\n🏁 Training completed in {total_training_time_main_final_run:.1f} seconds ({total_training_time_main_final_run/60:.1f} min) using LLM+Fusion method with Carbon Awareness and Composite Score.")


if __name__ == '__main__':
    main()