# LLMScheduler: plug-and-play replacement for HGNN module
# 
# Requirements:
#   - Python 3.8+
#   - PyTorch >= 2.0.0
#       pip install --upgrade torch
#   - sentence-transformers
#       pip install sentence-transformers
#
# Pretrained model:
#   Use "all-MiniLM-L6-v2" from sentence-transformers
#
# Example instantiation:
#   Create scheduler with just (model_name, hidden_dim, device):
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   sched = LLMScheduler(
#       model_name="all-MiniLM-L6-v2",
#       hidden_dim=64,
#       device=device
#   ).to(device)
#
# Example test harness snippet:
#   # 3) Run a single pass
#   with torch.no_grad():
#       probs, steps = sched.get_action_prob(state)
#       print("Action probabilities:", probs)  # shape [B, N_m]
#       actions = sched.act(state, memories=None, flag_sample=False)
#       print("Chosen action triple (ope,mac,job):", actions)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer
import copy

class LLMScheduler(nn.Module):
    """
    Low-latency scheduler using pretrained SentenceTransformer embeddings.
    """
    def __init__(self, model_name: str, hidden_dim: int, device: torch.device):
        super().__init__()
        # Load sentence-transformer
        self.encoder = SentenceTransformer(model_name, device=device)
        self.device = device
        embed_dim = self.encoder.get_sentence_embedding_dimension()
        # Actor: per (job,machine) score
        self.actor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        # Critic: optional state value
        self.critic = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def embed_nodes(self, descriptions: list[str]) -> torch.Tensor:
        return self.encoder.encode(descriptions, convert_to_tensor=True)

    def linearize_graph(self, raw_opes, raw_mas, proc_time):
        B, N_op, _ = raw_opes.size()
        _, N_m,  _ = raw_mas.size()
        ops_desc, mas_desc = [], []
        for b in range(B):
            for i in range(N_op):
                feats = raw_opes[b,i].tolist(); times = proc_time[b,i].tolist()
                ops_desc.append(f"Op{i}:f={feats};t={times}")
            for k in range(N_m):
                feats = raw_mas[b,k].tolist()
                mas_desc.append(f"Ma{k}:f={feats}")
        return ops_desc, mas_desc
    

    ### From original HGNN scheduler ###
    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''
    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)
    ### From original HGNN scheduler ###

    def get_action_prob(self, state, memories=None, flag_sample=False, flag_train=False):
        idx         = state.batch_idxes                            # [B]
        batch_adj   = state.ope_ma_adj_batch[idx]                  # [B, max_ops, N_m]
        B, max_ops, N_m = batch_adj.shape

        # 1) Pull out the raw features for this batch slice
        raw_opes    = state.feat_opes_batch.transpose(1, 2)[idx]   # [B, max_ops, F_op]
        raw_mas     = state.feat_mas_batch.transpose(1, 2)[idx]    # [B, N_m,    F_m]
        proc_time   = state.proc_times_batch[idx]                  # [B, max_ops, N_m]
        nums_opes   = state.nums_opes_batch[idx]                   # [B] (used only if you do per‐instance norm)

        # 2) Normalize exactly as you did in HGNN
        norm_opes, norm_mas, norm_proc = self.get_normalized(
            raw_opes, raw_mas, proc_time, idx, nums_opes, flag_sample, flag_train
        )

        # 3) Linearize + embed every op & every machine
        op_texts, ma_texts = [], []
        for b in range(B):
            for i in range(max_ops):
                f = raw_opes[b, i].cpu().tolist()
                t = proc_time[b, i].cpu().tolist()
                op_texts.append(f"O_{b}_{i}:f={f};t={t}")
            for k in range(N_m):
                f = raw_mas[b, k].cpu().tolist()
                ma_texts.append(f"M_{b}_{k}:f={f}")

        with torch.no_grad():
            op_emb = self.encoder.encode(op_texts, convert_to_tensor=True).to(self.device)
            ma_emb = self.encoder.encode(ma_texts, convert_to_tensor=True).to(self.device)

        # 4) reshape embeddings back to [B, max_ops, D] and [B, N_m, D]
        D    = op_emb.size(-1)
        h_op = op_emb.view(B, max_ops, D)
        h_ma = ma_emb.view(B, N_m, D)

        # 5) compute each job’s current head-op index
        step = torch.minimum(state.ope_step_batch, state.end_ope_biases_batch)[idx]
        if step.dim() == 1:
            G = 1
            step = step.unsqueeze(-1)
        else:
            G = step.size(1)

        # 6) build full [B, G, N_m] eligibility mask
        #    a) gather adj at head-op positions
        stg        = step.unsqueeze(-1).expand(-1, -1, N_m)          # [B, G, N_m]
        adj_expand = batch_adj.unsqueeze(1).expand(-1, G, -1, -1)    # [B, G, max_ops, N_m]
        eligible_proc = adj_expand.gather(2, stg.unsqueeze(2)).squeeze(2).bool()

        #    b) mask out busy machines & jobs
        ma_busy  = state.mask_ma_procing_batch[idx].unsqueeze(1).expand(-1, G, -1)
        job_busy = (state.mask_job_procing_batch[idx] | state.mask_job_finish_batch[idx])\
                    .unsqueeze(-1).expand(-1, -1, N_m)
        eligible = eligible_proc & ~ma_busy & ~job_busy

        # 7) gather the head-op embeddings per job
        hop_ex = h_op.unsqueeze(1).expand(-1, G, -1, -1)                                  # [B, G, max_ops, D]
        h_cur  = hop_ex.gather(2, step.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D))\
                    .squeeze(2)                                                         # [B, G, D]

        # 8) form pair embeddings [B, G, N_m, 2D] then flatten to [B, G*N_m, 2D]
        h_cur_ex = h_cur.unsqueeze(2).expand(-1, -1, N_m, -1)
        h_ma_ex  = h_ma.unsqueeze(1).expand(-1, G, -1, -1)
        pair_emb = torch.cat([h_cur_ex, h_ma_ex], dim=-1).view(B, G * N_m, 2 * D)

        # 9) score + mask + softmax → probs [B, G*N_m]
        mask_f = eligible.view(B, G * N_m)
        scores = self.actor(pair_emb).squeeze(-1)
        scores[~mask_f] = float("-inf")
        probs  = F.softmax(scores, dim=1)

        # 10) save everything PPO.update() expects
        if flag_train:
            memories.ope_ma_adj.append(batch_adj.clone())
            memories.ope_pre_adj.append(state.ope_pre_adj_batch[idx].clone())
            memories.ope_sub_adj.append(state.ope_sub_adj_batch[idx].clone())
            memories.batch_idxes.append(idx.clone())

            memories.raw_opes.append(norm_opes.clone())
            memories.raw_mas.append(norm_mas.clone())
            memories.proc_time.append(norm_proc.clone())
            memories.nums_opes.append(nums_opes.clone())

            # build a 3D gather index exactly like HGNN did (but only size-1 in the last dim)
            jobs_gather = step.unsqueeze(-1)  # [B, G, 1]

            memories.jobs_gather.append(jobs_gather.clone())
            memories.eligible.append(eligible.clone())

        return probs, step

    def act(self, state, memories, dones=None, flag_sample=True, flag_train=True):
        # 1) get logits and current step matrix
        probs, step = self.get_action_prob(state, memories, flag_sample, flag_train)
        B, flatA = probs.shape    # flatA = G * N_m

        # 2) recover number of jobs G and machines N_m
        if step.dim() > 1:
            G = step.size(1)
        else:
            G = 1
            step = step.unsqueeze(1)
        N_m = flatA // G

        # 3) build distribution
        dist = Categorical(probs)

        # 4) sample or greedy
        if flag_sample:
            idx_flat = dist.sample()    # [B]
        else:
            idx_flat = probs.argmax(dim=1)

        # ——— STORE IN MEMORIES HERE ———
        if flag_train:
            # store the log-prob of the chosen action
            memories.logprobs.append(dist.log_prob(idx_flat).clone())
            # store the raw flat action index
            memories.action_indexes.append(idx_flat.clone())

        # 5) decode job & machine by divmod
        job_idx = (idx_flat // N_m).long()  # [B]
        mac_idx = (idx_flat %  N_m).long()  # [B]

        # 6) lookup the op index for each chosen job
        op_idx = step.gather(1, job_idx.unsqueeze(1)).squeeze(1)  # [B]

        # 7) stack into (3, B) as (ope, mac, job)
        return torch.stack([op_idx, mac_idx, job_idx], dim=0)



    def evaluate(self,
                 ope_ma_adj: torch.Tensor,
                 ope_pre_adj: torch.Tensor,
                 ope_sub_adj: torch.Tensor,
                 raw_opes: torch.Tensor,
                 raw_mas: torch.Tensor,
                 proc_time: torch.Tensor,
                 jobs_gather: torch.Tensor,
                 eligible: torch.Tensor,
                 action_envs: torch.Tensor):
        """
        Called by PPO.update:
        - ope_ma_adj, ope_pre_adj, ope_sub_adj: [B, max_ops, N_m]
        - raw_opes:    [B, max_ops, F_op]
        - raw_mas:     [B, N_m,    F_m]
        - proc_time:   [B, max_ops, N_m]
        - jobs_gather: [B, G, 1]   (gather indices for each job)
        - eligible:    [B, G, N_m] eligibility mask
        - action_envs: [B]         flat action indexes sampled/taken
        Returns (action_logprobs, state_values, dist_entropy), each [B].
        """
        B = ope_ma_adj.size(0)
        max_ops = raw_opes.size(1)
        N_m     = raw_mas.size(1)

        # 1) Build string prompts (must mirror get_action_prob)
        op_texts, ma_texts = [], []
        for b in range(B):
            for i in range(max_ops):
                feats = raw_opes[b, i].cpu().tolist()
                times = proc_time[b, i].cpu().tolist()
                op_texts.append(f"O_{b}_{i}: feats={feats}; times={times}")
            for k in range(N_m):
                feats = raw_mas[b, k].cpu().tolist()
                ma_texts.append(f"M_{b}_{k}: feats={feats}")

        # 2) Embed with the same encoder
        with torch.no_grad():
            op_emb = self.encoder.encode(op_texts, convert_to_tensor=True).to(self.device)  # [B*max_ops, D]
            ma_emb = self.encoder.encode(ma_texts, convert_to_tensor=True).to(self.device)  # [B*N_m,    D]

        D     = op_emb.size(-1)
        h_op  = op_emb.view(B, max_ops, D)  # [B, max_ops, D]
        h_ma  = ma_emb.view(B, N_m,     D)  # [B, N_m,     D]

        # 3) Number of jobs G
        G = jobs_gather.size(1)

        # 4) Gather head-op embedding per job: [B, G, D]
        h_cur = h_op.gather(1, jobs_gather.expand(-1, -1, D))

        # 5) Build pair embeddings [B, G, N_m, 2D] → [B, G*N_m, 2D]
        h_cur_ex = h_cur.unsqueeze(2).expand(-1, -1, N_m, -1)
        h_ma_ex  = h_ma.unsqueeze(1).expand(-1, G, -1, -1)
        pair_emb = torch.cat([h_cur_ex, h_ma_ex], dim=-1).view(B, G * N_m, 2 * D)

        # 6) Score & mask
        scores     = self.actor(pair_emb).squeeze(-1)      # [B, G*N_m]
        mask_flat  = eligible.view(B, G * N_m)             # [B, G*N_m]
        scores[~mask_flat] = float("-inf")
        action_probs = F.softmax(scores, dim=1)

        # 7) Log‐probs of the *taken* actions
        dist           = Categorical(action_probs)
        action_logprobs= dist.log_prob(action_envs)       # [B]

        # 8) Critic: pool op & ma embeddings into a single state vector
        op_pooled      = h_op.mean(dim=1)                  # [B, D]
        ma_pooled      = h_ma.mean(dim=1)                  # [B, D]
        state_repr     = torch.cat([op_pooled, ma_pooled], dim=1)  # [B, 2D]
        state_values   = self.critic(state_repr).squeeze(-1)       # [B]

        state_values = state_values.to(torch.float64) # cast to float64 to follow rewards type

        # 9) Entropy bonus
        dist_entropy   = dist.entropy()                    # [B]

        return action_logprobs, state_values, dist_entropy

