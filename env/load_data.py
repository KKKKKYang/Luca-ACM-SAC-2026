import torch
import numpy as np

def load_fjs(lines, num_mas, num_opes):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    nums_ope = []  # A list of the number of operations for each job
    opes_appertain = np.array([])
    num_ope_biases = []  # The id of the first operation of each job
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line is "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            nums_ope.append(num_ope)
            # nums_option = np.concatenate((nums_option, num_option))
            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope)*(flag-1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes-opes_appertain.size)))
    #print("lines", lines, "num_mas", num_mas, "num_opes", num_opes)
    # print("-----------------------load_fjs--------------------------")
    # print("num_mas: ", num_mas, "num_opes: ", num_opes)
    # print("matrix_proc_time: ", matrix_proc_time[0])
    # print("matrix_ope_ma_adj: ", matrix_ope_ma_adj[0])
    # print("matrix_pre_proc: ", matrix_pre_proc[0])
    # print("matrix_pre_proc.t(): ", matrix_pre_proc.t()[0])


    # print("opes_appertain: ", opes_appertain[0])
    # print("num_ope_biases: ", num_ope_biases[0])
    # print("nums_ope: ", nums_ope[0])
    # print("matrix_cal_cumul: ", matrix_cal_cumul[0])
    # #shape
    # print("matrix_proc_time_shape: ", matrix_proc_time.shape)
    # print("matrix_ope_ma_adj_shape: ", matrix_ope_ma_adj.shape)
    # print("matrix_pre_proc_shape: ", matrix_pre_proc.shape)
    # print("matrix_pre_proc.t()_shape: ", matrix_pre_proc.t().shape) 
    # print("num_ope_biases_shape: ", len(num_ope_biases))    
    # print("nums_ope_shape: ", len(nums_ope))
    # print("matrix_cal_cumul_shape: ", matrix_cal_cumul.shape)

    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul

def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    # print("-----------------------nums_detec--------------------------")
    # print("lines", lines)
    # #print("line_split", line_split)
    # print("lines_shape", len(lines), "line_split_shape", len(line_split))
    # print("num_jobs: ", num_jobs, "num_mas: ", num_mas, "num_opes: ", num_opes)
    return num_jobs, num_mas, num_opes

def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope-1:
                matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope+num_ope_bias-1] = 1
                matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1]+vector
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    # print("-----------------------edge_detec--------------------------")
    # print('line:', line)
    # #print('line_split:', line_split)
    # print('num_ope:', num_ope, 'num_option:', num_option)
    # print('matrix_proc_time:', matrix_proc_time[0])
    return num_ope