import numpy as np
import time
import gpytorch
import torch
import logging
from model import ExactGPModel, MultitaskGPModel
from TreeNode import TreeNode
from sklearn.cluster import KMeans
from cvxopt import matrix
from cvxopt.solvers import qp
from matplotlib import pyplot as plt

def is_multi_start(algo):
    multi_start_algos = ["mbobyqa", "mlbfgsb", "mneldermead", "mpybobyqa", "mpyneldermead", "mpyVTS"]
    flag = False
    for name in multi_start_algos:
        if name in algo:
            flag = True
    return flag

def is_reinitialize_start(algo):
    multi_start_algos = ["mpybobyqa", "mpyVTS"]
    flag = False
    for name in multi_start_algos:
        if name in algo:
            flag = True
    return flag

def split_node(nodes, in_dim):
    split_list = [node.is_splittable() for node in nodes]
    while(any(split_list)):
        nodeid = split_list.index(True)
        lchild_data, rchild_data = nodes[nodeid].split()
        if len(lchild_data) > 0 and len(rchild_data) > 0:
            lchild = TreeNode(lchild_data, in_dim, max_leaf_size=2*in_dim)
            rchild = TreeNode(rchild_data, in_dim, max_leaf_size=2*in_dim)
            nodes.append(lchild)
            nodes.append(rchild)
        elif len(lchild_data) > 0:
            lchild = TreeNode(lchild_data, in_dim, max_leaf_size=2*in_dim, no_split=True)
            nodes.append(lchild)
        elif len(rchild_data) > 0:
            rchild = TreeNode(rchild_data, in_dim, max_leaf_size=2*in_dim, no_split=True)
            nodes.append(rchild)
        else: # impossible
            pass
        del nodes[nodeid]
        split_list = [node.is_splittable() for node in nodes]
    return nodes

# def get_leaf_node(root_node, in_dim):
#     node = root_node
#     while(node.is_splittable()):
#         lchild_data, rchild_data = node.split()
#         lchild_sorted_data = sorted(lchild_data, key=lambda data:data['value'])
#         rchild_sorted_data = sorted(rchild_data, key=lambda data:data['value'])
#         lchild_best_data = lchild_sorted_data[0]
#         rchild_best_data = rchild_sorted_data[0]
#         if lchild_best_data['value'] <= rchild_best_data['value']:
#             lchild = TreeNode(lchild_sorted_data, in_dim, lchild_best_data, max_leaf_size=2*in_dim)
#             node = lchild
#         else:
#             rchild = TreeNode(rchild_sorted_data, in_dim, rchild_best_data, max_leaf_size=2*in_dim)
#             node = rchild
#     return node

def fun_feedback(algo_list, num_worker_list, pconn_list, in_dim, budget, pass_percent, pass_divide, seed, rand_init, ref_start, func_sleep, update_num_workers=True, using_portfolio=True):
    # LOG_FORMAT = "%(asctime)s - %(levelname)s\n%(pathname)s - %(lineno)s - %(funcName)s\n%(message)s\n"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(lineno)s\n%(message)s\n"
    logging.basicConfig(filename='feedback.log', level=logging.INFO, format=LOG_FORMAT)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    save_fig = False
    device = 'cpu'
    cur_budget = 0
    num_worker_dict = dict(zip(algo_list, num_worker_list))
    temp_num_worker_dict = dict(zip(algo_list, num_worker_list))
    best_dict = dict.fromkeys(algo_list, None)
    best_one = np.inf
    worst_one = -np.inf
    best_record = None
    allrecords = []
    newrecords = []
    tot_num_workers = sum(num_worker_list)

    pass_tolerance = max(in_dim, int(budget/pass_divide))
    pass_round = max(int(pass_tolerance/5),2)
    round_cnt = 0
    restart_flag = False
    restart_algo_list = []
    leaf_nodes = None
    init_data = []
    inits = []

    tol_cnt_dict = dict.fromkeys(algo_list, 0)
    model_begin_dict = dict.fromkeys(algo_list, (0,0))
    model_dict = dict.fromkeys(algo_list, None) # for not using_portfolio 
    active_budget_dict = dict.fromkeys(algo_list, 0)
    for algo in algo_list:
        if is_multi_start(algo):
            active_budget_dict[algo] = num_worker_dict[algo]*4*in_dim
        else:
            active_budget_dict[algo] = 4*in_dim
    
    poor_cnt_dict = dict.fromkeys(algo_list, 0)
    terminate_algo_list = []
    no_terminate_algo_list = algo_list
    terminate_budget_dict = dict.fromkeys(algo_list, None)
    done_terminate_algos = []
    post_recv_algos = []
        

    def train_multitask_model(for_multi_mean_list, device):
        #nonlocal algo_list
        nonlocal no_terminate_algo_list
        assert len(no_terminate_algo_list) == len(for_multi_mean_list)
        train_y = torch.stack(
            [
                ys.to(device).type(torch.float32)
                for ys in for_multi_mean_list
            ], dim=-1
        )
        train_x = torch.linspace(0,1,100).to(device).type(torch.float32)*1e1
        model = MultitaskGPModel(
            train_x,
            train_y, 
            num_tasks=len(for_multi_mean_list),
            exp=False
            ).to(device)
        try:
            model.train()
            model.train_hypers(train_iter=200) 
        except RuntimeError as e:
            logging.info(repr(e))
            logging.info("in train_multitask_model, train_x {}".format(train_x))
            logging.info("in train_multitask_model, train_y {}".format(train_y))
            logging.info("try to train the multi-task GP model again")
            train_x[0] = 0.0001
            with gpytorch.settings.cholesky_jitter(1e-1):
                model = MultitaskGPModel(
                    train_x,
                    train_y, 
                    num_tasks=len(no_terminate_algo_list),
                    normal=False,
                    exp=False
                    ).to(device)
                model.train()
                model.train_hypers(train_iter=200) 
        return model 

    def train_a_model(y_list, algo, budget, cur_budget, device):
        nonlocal model_begin_dict
        nonlocal in_dim
        beginInfo = model_begin_dict[algo]
        begin_budget = beginInfo[0]
        begin_index = beginInfo[1]
        step = 1
        point_num = len(y_list) - begin_index
        # step = (len(y_list) - begin_index) // (cur_budget - begin_budget)
        # point_num = cur_budget - begin_budget
        clip_num = 1000
        if point_num > clip_num:
            step = max(1, int(point_num//clip_num))
            begin_budget = int(cur_budget - (cur_budget-begin_budget)*step*clip_num/point_num)
            begin_index = len(y_list) - step*clip_num
            logging.info("point_num clipped, begin_budget {}, begin_index {}".format(begin_budget, begin_index))
            #model_begin_dict[algo] = (begin_budget, begin_index)
            point_num = clip_num
        #train_y = torch.Tensor(y_list[begin_index::step]).to(device).type(torch.float32)
        train_y = torch.Tensor(y_list[:-point_num*step-1:-step][::-1]).to(device).type(torch.float32)
        train_x = torch.linspace(0,cur_budget-begin_budget,len(train_y)).to(device).type(torch.float32)/budget*1e6
        model = ExactGPModel(
            train_x,
            train_y, 
            ).to(device)
        try:
            model.train()
            model.train_hypers(train_iter=100) 
        except RuntimeError as e:
            logging.info(repr(e))
            logging.info("in train_a_model, train_y {}".format(train_y))
            logging.info("in train_a_model, train_x {}".format(train_x))
            logging.info("try to train the GP model again")
            train_x[0] = 0.0001
            model = ExactGPModel(
                train_x,
                train_y, 
                ).to(device)
            model.train()
            model.train_hypers(train_iter=100) 
        return model

    while(cur_budget < budget):
        cur_budget += 1
        logging.info("========== cur_budget: {} ==========".format(cur_budget))
        no_terminate_algo_list = [algo for algo in algo_list
                                  if algo not in terminate_algo_list]
        logging.info(
            "no_terminate_algo_list {}, terminate_algo_list {}\n num_worker_dict {}".format(
                no_terminate_algo_list, terminate_algo_list, num_worker_dict
            )
        )
        post_recv_algos = []
        conn_timeout = func_sleep * 1.5
        for algo, pconn in zip(algo_list, pconn_list):
            if algo in terminate_algo_list: # Need not recv 
                logging.info("algo {} has been terminated".format(algo))
                continue
            logging.info("waiting for algo {} conn".format(algo))
            start_time = time.time()
            poll_flag = True
            if restart_flag or pconn.poll(conn_timeout):
                records = pconn.recv()
            else:
                poll_flag = False
                records = num_worker_dict[algo] * ['STOP']
                post_recv_algos.append(algo)
            # print("algo {} records {}".format(algo, records))
            for record in records:
                # Update allrecords
                if record != 'STOP' and \
                    all(
                    [np.any(np.abs(record['candidate_value']-i['candidate_value'])>=1e-3) 
                     for i in allrecords]
                        ):
                    allrecords.append(record)
                    newrecords.append(record)
                # Update best_dict
                if record=='STOP' and best_dict[algo] is None:
                    best_dict[algo] = [dict(candidate_value=rand_init, value=ref_start)]
                elif best_dict[algo] is None:
                    best_dict[algo] = [record]
                elif record == 'STOP':
                    best_dict[algo].append(best_dict[algo][-1])
                elif record['value'] < best_dict[algo][-1]['value']:
                    best_dict[algo].append(record)
                else:
                    best_dict[algo].append(best_dict[algo][-1])
                # Update best_one
                if record != 'STOP' and record['value'] < best_one:
                    best_one = record['value']
                    best_record = record
                # Update worst_one
                if record != 'STOP' and record['value'] > worst_one:
                    worst_one = record['value']
            logging.info("len allrecords {}".format(len(allrecords)))
            logging.info("len newrecords {}".format(len(newrecords)))
            logging.info("algo {} records received, poll_flag {}, conn_timeout {}".format(algo, poll_flag, conn_timeout))
            logging.info("post_recv_algos {}".format(post_recv_algos))
            end_time = time.time()
            delta_time = end_time - start_time
            conn_timeout -= delta_time
            conn_timeout = max(0, conn_timeout)

        # Execute the strategy of restart and reallocation
        round_cnt = (round_cnt + 1) % pass_round

        # Prepare send_list
        send_list = []
        for algo, bestline in best_dict.items():
            logging.info("algo {} len bestline {}".format(algo, len(bestline)))
            # Restart case, round_cnt == 1
            if restart_flag:
                if algo in terminate_algo_list:
                    logging.info("algo {} has been terminated, num_workers {}".format(algo, num_worker_dict[algo]))
                    send_list.append(False)
                elif algo in restart_algo_list:
                    logging.info("algo {} restarted, num_workers {}".format(algo, num_worker_dict[algo]))
                    send_list.append(True)
                    # Update tolerance counts, active budgets, model_begin_dict and (best_dict)
                    tol_cnt_dict[algo] = 0
                    if is_reinitialize_start(algo):
                        active_plus = num_worker_dict[algo] * in_dim
                    elif is_multi_start(algo):
                        active_plus = num_worker_dict[algo] * 2 * in_dim
                    else:
                        active_plus = in_dim
                    active_budget_dict[algo] = len(best_dict[algo]) + active_plus
                    if is_multi_start(algo):
                        model_begin_dict[algo] = (cur_budget, len(best_dict[algo]))
                else:
                    logging.info("algo {} gets rid of restart, num_workers {}".format(algo, num_worker_dict[algo]))
                    send_list.append(False)
            # Normel case
            else:
                send_list.append(False)
        # print("send_list {}".format(send_list))
        logging.info("send list {}".format(send_list))
        for algo, pconn, send_flag in zip(algo_list, pconn_list, send_list):
            if send_flag:
                logging.info("algo {} send_flag {}".format(algo, send_flag))
                try:
                    if is_reinitialize_start(algo):
                        send_workers = num_worker_dict[algo]
                        while(len(init_data) < send_workers): # in case of update_workers False
                            init_data += init_data
                        pconn.send((init_data[:send_workers], send_workers))
                    else:
                        send_workers = num_worker_dict[algo]
                        while(len(inits) < send_workers): # in case of update_workers False
                            inits += inits
                        pconn.send((inits[:send_workers], send_workers))
                except Exception as e:
                    logging.error(repr(e))
                    raise
            else:
                logging.info("algo {} send_flag {}".format(algo, send_flag))
                try:
                    # Do not sent for post_recv_algos because they have not sent records, 
                    # otherwise they will get redundant information to receive
                    if algo in post_recv_algos:
                        logging.info("{} in post_recv_algos, do not send".format(algo))
                        pass 
                    elif algo in done_terminate_algos:
                        logging.info("{} in done_terminate_algos, do not send".format(algo))
                        pass
                    elif algo in terminate_algo_list:
                        logging.info("send information to terminate algo {}".format(algo))
                        pconn.send((None, 0)) # The corresponding algo will be stopped
                        done_terminate_algos.append(algo)
                    else:
                        logging.info("send double None information for algo {}".format(algo))
                        pconn.send((None, None))
                except Exception as e:
                    logging.error(repr(e))
                    raise

        # Update multitask_model, which is for updating tol_cnt_dict then, every pass_round
        restart_flag = False
        if round_cnt or cur_budget > budget-in_dim or cur_budget < 4*in_dim:
            logging.info("cur_budget is {}, skip training".format(cur_budget))
        else:
            logging.info("cur_budget is {}, start to train model_dict".format(cur_budget))
            try:
                model_dict = dict.fromkeys(no_terminate_algo_list, None)
                for algo in no_terminate_algo_list:
                    model_dict[algo] = train_a_model(
                                        [record['value'] for record in best_dict[algo]], 
                                        algo, 
                                        budget, 
                                        cur_budget,
                                        device
                                        )
            except Exception as e:
                logging.error(repr(e))
                raise
            # Using model_dict 
            # Prepare data (end_mean, end_stddev, PI_list)
            end_mean = []
            end_stddev = []
            PI_list = []
            if save_fig:
                cur_line_list = []
                predict_x_list = []
                predict_mean_list = []
                predict_upper_list = []
                predict_lower_list = []
            if using_portfolio:
                for_multi_mean_list = []
            for algo in algo_list:
                if algo in model_dict.keys(): # no_terminated_algo_list
                    model = model_dict[algo]
                    model.eval()
                    begin_budget = model_begin_dict[algo][0]
                    temp_budget = min(cur_budget+(cur_budget-begin_budget), budget)
                    test_x = torch.arange(0,temp_budget-begin_budget).to(device).type(torch.float32)/budget*1e6
                    predict_x = torch.arange(
                        begin_budget, 
                        temp_budget
                        ).to(device).type(torch.float32)/budget*1e6
                    try:
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            predict_preds = model.posterior(test_x)
                            predict_mean = model.y_retransform(predict_preds.mean)
                            predict_stddev= model.sigma_retransform(predict_preds.stddev)
                            #arg_point = torch.argmin(predict_mean)
                            arg_point = torch.where(predict_mean == torch.min(predict_mean))[0][-1]
                            end_mean.append(predict_mean[arg_point])
                            end_stddev.append(predict_stddev[arg_point])
                            PI_point = predict_x[arg_point]
                            logging.info("algo {} PI_point {}".format(algo, PI_point.item()/1e6))
                            PI_list.append(
                                model.calc_PI(PI_point.unsqueeze(0), torch.Tensor([best_one])).item()
                            )
                    except Exception as e:
                        logging.info(repr(e))
                        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-1):
                            predict_preds = model.posterior(test_x)
                            predict_mean = model.y_retransform(predict_preds.mean)
                            predict_stddev= model.sigma_retransform(predict_preds.stddev)
                            arg_point = torch.where(predict_mean == torch.min(predict_mean))[0][-1]
                            end_mean.append(predict_mean[arg_point])
                            end_stddev.append(predict_stddev[arg_point])
                            PI_point = predict_x[arg_point]
                            logging.info("algo {} PI_point {}".format(algo, PI_point.item()/1e6))
                            PI_list.append(
                                model.calc_PI(PI_point.unsqueeze(0), torch.Tensor([best_one])).item()
                            )
                    if save_fig:
                        cur_line = [record['value'] for record in best_dict[algo]]
                        # cur_line = []
                        # for record in best_dict[algo]:
                        #     if record['value'] != 'STOP':
                        #         cur_line.append(record['value'])
                        #     else:
                        #         cur_line.append(cur_line[-1])
                        predict_stddev = model.sigma_retransform(predict_preds.stddev)
                        predict_lower = predict_mean.sub(predict_stddev)
                        predict_upper = predict_mean.add(predict_stddev)
                        cur_line_list.append(cur_line)
                        predict_x_list.append(predict_x[:arg_point+1].cpu().numpy()/1e6)
                        predict_mean_list.append(predict_mean[:arg_point+1].cpu().numpy())
                        predict_lower_list.append(predict_lower[:arg_point+1].cpu().numpy())
                        predict_upper_list.append(predict_upper[:arg_point+1].cpu().numpy())
                    if using_portfolio:
                        for_multi_test_x = torch.linspace(0,test_x[arg_point], 100).to(device).type(torch.float32)
                        for_multi_preds = model.posterior(for_multi_test_x)
                        for_multi_mean = model.y_retransform(for_multi_preds.mean)
                        for_multi_mean_list.append(for_multi_mean.detach())
                else:
                    if save_fig:
                        cur_line = [record['value'] for record in best_dict[algo]]
                        # predict_lower = predict_mean.sub(predict_stddev)
                        # predict_upper = predict_mean.add(predict_stddev)
                        cur_line_list.append(cur_line)
                        predict_x_list.append(None)
                        predict_mean_list.append(None)
                        predict_lower_list.append(None)
                        predict_upper_list.append(None)
            end_mean = torch.Tensor(end_mean)
            end_stddev = torch.Tensor(end_stddev)
            logging.info("cur_budget is {}, end_mean {}, end_stddev {}, PI_list {}".format(
                            cur_budget, end_mean, end_stddev, PI_list))
            if using_portfolio: # Prepare data (end_covar)
                logging.info("cur_budget is {}, start to train multitask_model for inter covariance".format(cur_budget))
                try:
                    multitask_model = train_multitask_model(
                                        for_multi_mean_list,
                                        device
                                        )
                except Exception as e:
                    logging.error(repr(e))
                    raise
                # Using multitask_model
                multitask_model.eval()
                test_x = torch.Tensor([1]).to(device).type(torch.float32)*1e1
                try:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        preds = multitask_model.posterior(test_x)
                        covar_multi = preds.covariance_matrix
                        end_covar = multitask_model.sigma_retransform(
                            covar_multi[-len(no_terminate_algo_list):,-len(no_terminate_algo_list):]
                        )
                except Exception as e:
                    logging.info(repr(e))
                    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-1):
                        preds = multitask_model.posterior(test_x)
                        covar_multi = preds.covariance_matrix
                        end_covar = multitask_model.sigma_retransform(
                            covar_multi[-len(no_terminate_algo_list):,-len(no_terminate_algo_list):]
                        )
                logging.info("cur_budget is {}, end_covar {}".format(
                                cur_budget, end_covar))
            # Using data (PI_list)
            try:
                # Update tol_cnt_dict wether update_num_workers or not
                for index, algo in enumerate(no_terminate_algo_list):
                    if best_dict[algo][-1]['value'] < best_one+0.001*(worst_one-best_one):
                        logging.info("algo {} competitive for best_one, tol_cnt keep zero".format(algo))
                        tol_cnt_dict[algo] = 0
                    elif PI_list[index] < 1-pass_percent:
                        if len(best_dict[algo])>active_budget_dict[algo]: 
                            logging.info("algo {} tol_cnt added one".format(algo))
                            tol_cnt_dict[algo] += 1
                        else:
                            logging.info("algo {} is not active, tol_cnt keep zero".format(algo))
                            tol_cnt_dict[algo] = 0
                    else:
                        logging.info("algo {} tol_cnt keep zero".format(algo))
                        tol_cnt_dict[algo] = 0
                logging.info(
                    "cur_budget {}\ntol_cnt_dict{}".format(cur_budget, tol_cnt_dict)
                    )
                if np.any(np.array([i for i in tol_cnt_dict.values()]) >= 3):
                    # Restart algos in the next pass_round
                    logging.info("restart_flag is enabled")
                    restart_flag = True 
                    # Partition the evaluated data points and filter out promising regions
                    # for multi_start algos, making their num_workers limited
                    try:
                        if leaf_nodes is None:
                            root_node = TreeNode(allrecords, in_dim, max_leaf_size=2*in_dim)
                            to_split_nodes = [root_node]
                            new_records = []
                        else:
                            represents = np.array(
                                          [node.get_best()['candidate_value'] for node in leaf_nodes]
                                         )
                            for record in newrecords:
                                represent_id = np.argmin(
                                                np.sqrt(
                                                 np.sum(
                                                  np.square(represents - record['candidate_value']), 
                                                  axis=1
                                                 )
                                                )
                                               )
                                leaf_nodes[represent_id].add_data(record)
                            to_split_nodes = leaf_nodes
                            newrecords = []
                        leaf_nodes = split_node(to_split_nodes, in_dim)
                        logging.info("len leaf_nodes: {}".format(len(leaf_nodes)))
                        if worst_one  > best_one:
                            y_line = best_one + (worst_one-best_one)/5
                            init_nodes = [node for node in leaf_nodes 
                                        if node.get_best()['value'] < y_line]
                            init_nodes = sorted(init_nodes, key=lambda x:x.get_best()['value'])
                            logging.info("len init_nodes: {}".format(len(init_nodes)))
                        else:
                            init_nodes = sorted(leaf_nodes, key=lambda x:x.num_data)[0:1]
                            logging.info(
                                "worst_one equals to best_one, choose one node with the least data"
                                )
                        init_data = [node.provide_data() for node in init_nodes]
                        inits = [node.get_best() for node in init_nodes]
                        multi_lim = len(init_data)
                    except Exception as e:
                        logging.error(repr(e))
                        raise
                    # Get restart_algo_list and then check re_num_workers
                    restart_algo_list = []
                    mask_list = []
                    re_num_workers = tot_num_workers
                    temp_poor_algos = []
                    for algo in no_terminate_algo_list:
                        # Restart algos with large tol_cnt
                        # Note that not is_multi_start algos won't be truly restarted,
                        # but they are always appended because the num_workers may change 
                        # if update_num_workers
                        if tol_cnt_dict[algo] >= 3:
                            temp_poor_algos.append(algo)
                            logging.info("add poor restart algo...")
                            restart_algo_list.append(algo)
                            mask_list.append(True)
                            tol_cnt_dict[algo] = 0
                        elif not is_multi_start(algo):
                            logging.info("add fake restart algo...")
                            restart_algo_list.append(algo)
                            mask_list.append(True)
                        # If tol_cnt less than 3 and is_multi_start, 
                        # the algo won't be restarted nor reallocated 
                        else: 
                            mask_list.append(False)
                    if len(temp_poor_algos) < len(algo_list): # Avoid terminating all algos
                        for algo in temp_poor_algos:
                            poor_cnt_dict[algo] += 1
                    new_terminate_algo_list = []
                    for algo in temp_poor_algos:
                        # Avoid all restart algos being terminated
                        if len(new_terminate_algo_list) < len(restart_algo_list) - 1:
                            # For reinitialize_start algo tolerance is higher
                            # 1: no restart 2: one chance to restart 3: two chances to restart ...
                            if is_reinitialize_start(algo):
                                if poor_cnt_dict[algo] >= 3:
                                    logging.info("add reinitialize_start terminate algo ...")
                                    new_terminate_algo_list.append(algo)
                            elif poor_cnt_dict[algo] >= 2:
                                logging.info("add terminate algo ...")
                                new_terminate_algo_list.append(algo)
                            else:
                                pass
                    if update_num_workers:
                        for algo in new_terminate_algo_list:
                            terminate_algo_list.append(algo)
                            terminate_budget_dict[algo] = cur_budget
                    no_terminate_restart_list = [algo for algo in restart_algo_list
                                                 if algo not in terminate_algo_list]
                    re_num_workers = sum([num_worker_dict[algo] for algo in restart_algo_list])
                    # Every restart algo keep 1 if not to be terminated
                    re_num_workers -= len(no_terminate_restart_list)
                    mask_array = np.array(mask_list)
                    logging.info(
                        "restart_algo_list {}, no_terminate_restart_list {}, re_num_workers {}".format(
                            restart_algo_list, no_terminate_restart_list, re_num_workers
                        )
                    )
                    if not ((re_num_workers > 0 and update_num_workers and len(restart_algo_list)>=2)):
                        logging.info("restart but no reallocation")
                    # Reallocation among restart algos
                    # Some or even all these algos may be terminated
                    else:
                        logging.info("restart with reallocating num_workers")
                        no_terminate_workers = [num_worker_dict[algo] for algo in no_terminate_algo_list]
                        restart_cur_workers = [num_worker_dict[algo] for algo in restart_algo_list]
                        restart_cur_tot = sum(restart_cur_workers)
                        # Update temp_num_worker_dict according to the predictive performances
                        # Get perform_weight
                        if using_portfolio: # Using data (end_mean, end_stddev, end_covar)
                            portfolio_fail = False
                            try:
                                ## Portfolio
                                unit_array = np.array(restart_cur_workers)
                                unit2_array = unit_array[np.newaxis,:]
                                mean_array = end_mean.detach().cpu().numpy()[mask_array]
                                ref_one = best_one + (worst_one - best_one) * 0.3
                                reward_array = (ref_one - mean_array) / unit_array
                                reward_array = np.clip(reward_array, 0)
                                stddev_array = end_stddev.detach().cpu().numpy()[mask_array] / unit_array
                                covar_array = end_covar.detach().cpu().numpy()[mask_array,:][:,mask_array]
                                covar_array = covar_array / np.sqrt(unit2_array * unit2_array.T)
                                # update covar_array
                                covar_array = covar_array - np.diag(np.diag(covar_array)) + np.diag(stddev_array)
                                max_array = np.array(
                                            [min(1.0, multi_lim/restart_cur_tot) 
                                             if is_multi_start(algo) else 1.0
                                             for algo in restart_algo_list]
                                             )
                                risk_aversion = (budget - cur_budget)/budget * 4 + 1 # 5 -> 1
                                logging.info("portfolio reward: {} {}".format(reward_array, reward_array.dtype))
                                logging.info("portfolio covar: {} {}".format(covar_array, covar_array.dtype))
                                logging.info("portfolio max_lims: {} {}".format(max_array, max_array.dtype))
                                logging.info("portfolio risk_aversion: {}".format(risk_aversion))
                                c = 0 # no fees
                                T = 1 # turnover constraint
                                n = len(reward_array)
                                r = reward_array.astype(np.double)
                                r = matrix(np.block([r, -c * np.ones(2*n)]))
                                Q = covar_array.astype(np.double) * risk_aversion
                                Q = matrix(np.block([[Q, np.zeros((n,n)), np.zeros((n,n))], 
                                                     [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))], 
                                                     [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]]))
                                # A = matrix(np.ones(n)).T
                                A = matrix(np.block([[np.ones(n), c * np.ones(n), -c * np.ones(n)], 
                                                     [np.eye(n), np.eye(n), -np.eye(n)]]))
                                # b = matrix(1.0)
                                cur_x = np.array(restart_cur_workers)
                                cur_x = cur_x/restart_cur_tot
                                logging.info("portfolio cur_x: {}".format(cur_x))
                                b = matrix(np.block([1.0, cur_x]))
                                # G = matrix(-np.eye(n))
                                G = matrix(np.block([[-np.eye(3*n)],
                                                     [np.eye(3*n)],
                                                     [np.zeros(n), np.ones(2*n)]]))
                                # h = matrix(np.zeros(n))
                                h = matrix(np.block(
                                    [np.zeros(3*n), np.hstack([max_array, np.ones(2*n)]), T]
                                    ))
                                sol = qp(Q, -r, G, h, A, b)
                                perform_weight = sol['x'][:n]
                                logging.info("portfolio solution: {}".format(perform_weight))
                            except Exception as e:
                                logging.info(repr(e))
                                logging.info("portfolio failed, try to reallocate workers based on normalized PI")
                                portfolio_fail = True
                        if (not using_portfolio) or portfolio_fail: # Using data (PI_list)
                            ## Normalized unit PI
                            unit_list = [i/j for i,j in zip(PI_list, no_terminate_workers)]
                            unit_array = np.array(unit_list)
                            logging.info("unit_array {}".format(unit_array))
                            perform_array = unit_array[mask_array]
                            perform_sum = np.sum(perform_array)
                            if perform_sum > 1e-6:
                                perform_weight = perform_array/perform_sum
                            else:
                                perform_weight = np.array([1/len(perform_array)]*len(perform_array))
                            logging.info("normalized unit PI solution: {}".format(perform_weight))
                        # Update temp_num_worker_dict for restart algos, adding the kept 1
                        # For to-be-terminated algos the num_workers will be 0 later
                        realloc_temp_num_worker_dict = dict(
                            zip(restart_algo_list, re_num_workers * perform_weight + 1)
                        )

                        left_num_workers = re_num_workers + len(no_terminate_restart_list)
                        for algo in no_terminate_restart_list[:-1]:
                            if is_multi_start(algo):
                                num_reach = min(max(1,int(realloc_temp_num_worker_dict[algo])), multi_lim)
                                temp_num_worker_dict[algo] = num_reach
                                left_num_workers -= num_reach
                            else:
                                num_reach = max(1,int(realloc_temp_num_worker_dict[algo]))
                                temp_num_worker_dict[algo] = num_reach
                                left_num_workers -= num_reach
                        left_algo = no_terminate_restart_list[-1]
                        temp_num_worker_dict[left_algo] = left_num_workers

                        # Zero num_workers for to-be-terminated algos
                        for algo in terminate_algo_list:
                            temp_num_worker_dict[algo] = 0

                        # Since next pass_round restart, update num_worker_dict
                        num_worker_dict = temp_num_worker_dict
                logging.info(
                    "cur_budget {}\nnum_worker_dict {}".format(cur_budget, num_worker_dict)
                    )
                if save_fig:
                    _, ax1 = plt.subplots(1, 1, figsize=(16, 12), layout="constrained")
                    for algo, cur_line, x, mean, lower, upper in \
                        zip(algo_list,
                            cur_line_list, 
                            predict_x_list,
                            predict_mean_list,
                            predict_lower_list,
                            predict_upper_list):
                        if algo in no_terminate_algo_list:
                            logging.info("algo {} predict_mean {}".format(algo, mean))
                            ax1.plot(np.linspace(0,cur_budget,len(cur_line))/budget, cur_line, c='k')
                            ax1.plot(x, mean)
                            # Shade between the lower and upper confidence bounds
                            ax1.fill_between(x, lower, upper, alpha=0.5)
                            #ax1.legend(['Training data', 'Mean', 'Confidence'])
                        else:
                            ax1.plot(np.linspace(0,terminate_budget_dict[algo],len(cur_line))/budget, cur_line, c='k')
                            terminate_x = np.array(range(terminate_budget_dict[algo], cur_budget))/budget
                            terminate_y = [cur_line[-1]]*len(terminate_x)
                            ax1.plot(terminate_x, terminate_y, '--')
                            ax1.fill_between(terminate_x, terminate_y, terminate_y, alpha=0.1)
                    plt.savefig("plot_{}.png".format(cur_budget))
            except Exception as e:
                logging.error(repr(e))
                raise

    # End phase
    for algo, pconn in zip(algo_list, pconn_list):
        if algo in done_terminate_algos:
            logging.info("end phase, algo {} has been terminated".format(algo))
            continue
        else:
            if algo in post_recv_algos:
                logging.info("end phase, receive information from algo {}".format(algo))
                pconn.poll(timeout=None)
                end_records = pconn.recv()
                for record in end_records:
                    # Update best_one
                    if record != 'STOP' and record['value'] < best_one:
                        best_one = record['value']
                        best_record = record
            logging.info("end phase, send information to terminate algo {}".format(algo))
            pconn.send((None, 0)) # The corresponding algo will be stopped
            done_terminate_algos.append(algo)

    logging.info(
        "cur_budget {}\nfeedback is done".format(cur_budget)
        )

    return best_one
