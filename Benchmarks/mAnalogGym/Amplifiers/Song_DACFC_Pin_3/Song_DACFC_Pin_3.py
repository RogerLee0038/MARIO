import os
import shutil
import subprocess
# import pickle
# import time
import math
import numpy as np
import re

class Song_DACFC:
    def __init__(self, index=0):
        self.name = "Song_DACFC"
        self.suffix = ""
        self.index = index
        self.dir = os.path.dirname(__file__)
        assert os.path.exists(os.path.join(self.dir, "circuit"))
        # self.database = "TwoStageOpamp.pkl"
        self.mode = "spice" # or "ocean"
        self.del_folders = True

        # Design Variables
        ## DX = [('name',L,U,step,init,[discrete list]),....] if there is no discrete, do not write
        self.DX = [
            ('MOSFET_58_2_L_gmf1_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_58_2_M_gmf1_PMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_58_2_W_gmf1_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('CAPACITOR_1', 1.4e-13, 3.5e-12, 1e-15, 7e-13, 'NO'),
            ('MOSFET_0_8_L_BIASCM_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_0_8_M_BIASCM_PMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_0_8_W_BIASCM_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_10_1_L_gm2_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_10_1_M_gm2_PMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_10_1_W_gm2_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_11_1_L_gmf2_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_11_1_M_gmf2_PMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_11_1_W_gmf2_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_17_7_L_BIASCM_NMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_17_7_M_BIASCM_NMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_17_7_W_BIASCM_NMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_21_2_L_LOAD2_NMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_21_2_M_LOAD2_NMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_21_2_W_LOAD2_NMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_23_1_L_gm3_NMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_23_1_M_gm3_NMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_23_1_W_gm3_NMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_5_2_L_gma1_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_5_2_M_gma1_PMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_5_2_W_gma1_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_7_1_L_gma2_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_7_1_M_gma2_PMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_7_1_W_gma2_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_8_2_L_gm1_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('MOSFET_8_2_M_gm1_PMOS', 1.0, 20.0, 1, 4.0, 'NO'),
            ('MOSFET_8_2_W_gm1_PMOS', 0.5, 5.0, 0.1, 1.0, 'NO'),
            ('CAPACITOR_0', 3e-13, 7.5e-12, 1e-15, 1.5e-12, 'NO'),
            ('CURRENT_0_BIAS', 1e-06, 2.5e-05, 1e-07, 5e-06, 'NO'),
        ]

        self.in_dim = len(self.DX)
        self.real_init = np.array([dx[4] for dx in self.DX])
        self.real_lb = np.array([dx[1] for dx in self.DX])
        self.real_ub = np.array([dx[2] for dx in self.DX])
        self.init = (self.real_init-self.real_lb)/(self.real_ub - self.real_lb)

        self.run_file = "run_sim.sh"
        self.result_file = "result.po"

        self.perform_setting = {
        "weighted_fom": (None, None, None, None, "weighted_fom", 1e30, 10),
        }
        self.fom_setting = (1, None)

    def cal_fom(self, meas_dict):
        fom = meas_dict["weighted_fom"]
        return fom

    def write_param(self, dx_real_dict):
        if self.mode == "spice":
            with open("param", "w") as handler:
                for dx_name, dx_real in dx_real_dict.items():
                    handler.write(".param {}={}\n".format(dx_name, dx_real))
        elif self.mode == "ocean":
            handler.write('ocnxlSweepVar(\"' + str(dx_name) + '\" ' + '\"' + str(dx_real) + '\")\n')
        else:
            raise Exception("unknown self.mode")

    def extract_perf(self, file_name, perf):
        pattern_str = perf+'\s*=\s*([\d.eE+\-]+)'
        pattern = re.compile(pattern_str)
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                result = pattern.search(line)
                if result:
                    val = result.group(1)
                    return float(val)
            return False

    ##################################################################

    def set_name_suffix(self, suffix):
        self.suffix = suffix
        return self

    def __call__(self, x, realx=False, index=None):
        # while(os.path.exists("index_lock")):
        #     print("index is locked, wait for 1s")
        #     time.sleep(1)
        # open("index_lock", "a").close()
        # if os.path.exists("index.pkl"):
        #     with open("index.pkl", "rb") as fr:
        #         old_index = pickle.load(fr)
        #         tmp_index = old_index + 1
        # else:
        #     tmp_index = 0
        # with open("index.pkl", "wb") as fw:
        #     pickle.dump(tmp_index, fw)
        # os.remove("index_lock")
        if index is None: # sequentially
            tmp_index = self.index
            self.index += 1
        else: # parallel with index updated in global
            tmp_index = index
        tmp_dir = "{}_{}_{}".format(self.name, self.suffix, tmp_index)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        cwd = os.getcwd()
        shutil.copytree(
            os.path.join(self.dir, "circuit"), 
            os.path.join(cwd, tmp_dir)
        )
        print("{} is created, waiting for simulation".format(tmp_dir))
        os.chdir(tmp_dir)
        if not realx:
            x_01 = x
            dx_real_dict = self.dx_map(x_01)
        else:
            x_real = x
            x_name = [dx[0] for dx in self.DX]
            dx_real_dict = dict(zip(x_name,x_real))
        self.write_param(dx_real_dict)
        subprocess.Popen([self.run_file]).wait()
        print("{} simulation done".format(tmp_dir))
        meas_dict = self.read_meas(self.result_file)
        fom = self.cal_fom(meas_dict)
        cost = self.cal_cost(meas_dict, fom)
        print("{} get cost {}".format(tmp_dir, cost))
        os.chdir(cwd)
        # self.update_database(tmp_index, dx_real_dict, meas_dict, fom, cost)
        if self.del_folders:
            shutil.rmtree(tmp_dir)
        return cost

    def dx_map(self, x_01):
        dx_real_dict = {}
        for dx_tup, dx_01 in zip(self.DX, x_01):
            dx_name = dx_tup[0]
            dx_lb = dx_tup[1]
            dx_ub = dx_tup[2]
            dx_step = dx_tup[3]
            dx_real_range = dx_01*(dx_ub-dx_lb)
            plus = 1 if (dx_real_range%dx_step)/dx_step >= 0.5 else 0
            round_range = dx_real_range//dx_step*dx_step + plus*dx_step
            dx_real = round_range + dx_lb
            if dx_real > dx_ub:
                dx_real = dx_ub
            if dx_real < dx_lb:
                dx_real = dx_lb
            dx_real_dict[dx_name] = dx_real
        return dx_real_dict

    def read_meas(self, file_name):
        meas_dict = {}
        for perform_name, perform_tup in self.perform_setting.items():
            perform_value = self.extract_perf(file_name, perform_tup[4])
            if not perform_value:
                perform_value = perform_tup[5]
            meas_dict[perform_name] = perform_value
        return meas_dict

    def cal_cost(self, meas_dict, fom):
        cons_list = []
        for perform_name, perform_value in meas_dict.items():
            tup = self.perform_setting[perform_name]
            spec_weight= tup[-1] if tup[-1] else 1
            if "<" in tup:
                if tup[1] != 0:
                    cons_list.append(
                        (perform_value - tup[1])/abs(tup[1])*spec_weight
                    )
                else:
                    cons_list.append(
                        (2/(1+math.exp(-1*perform_value))-1)*spec_weight
                    )
            if ">" in tup:
                if tup[3] != 0:
                    cons_list.append(
                        -(perform_value - tup[3])/abs(tup[3])*spec_weight
                    )
                else:
                    cons_list.append(
                        -(2/(1+math.exp(-1*perform_value))+1)*spec_weight
                    )
            if ("<" not in tup) and (">" not in tup):
                continue
        cons_cost = sum([x if x>0 else 0 for x in cons_list])
        fom_weight = self.fom_setting[-1] if self.fom_setting[-1] else 1
        fom_cost = (fom - self.fom_setting[0])/abs(self.fom_setting[0])*fom_weight
        cost = cons_cost + fom_cost
        return cost

    # def update_database(self, index, dx_real_dict, meas_dict, fom, cost):
    #     # Prepare datas
    #     x_real = [dx_real for dx_real in dx_real_dict.values()]
    #     while(os.path.exists("database_lock")):
    #         print("database is locked, wait for 1s")
    #         time.sleep(1)
    #     open("database_lock", "a").close()
    #     if os.path.exists(self.database):
    #         with open(self.database, "rb") as fr:
    #             datas = pickle.load(fr)
    #             datas.append(dict(index=index, x_real=x_real, meas=meas_dict, fom=fom, cost=cost, time=time.time()))
    #     else:
    #         datas = [dict(index=index, x_real=x_real, meas=meas_dict, fom=fom, cost=cost, time=time.time())]
    #     with open(self.database, "wb") as fw:
    #         pickle.dump(datas, fw)
    #     os.remove("database_lock")

if __name__ == '__main__':
    three_stage_opamp = ThreeStageOpamp()
    three_stage_opamp(three_stage_opamp.real_init, realx=True)
    three_stage_opamp([0.5]*three_stage_opamp.in_dim)
