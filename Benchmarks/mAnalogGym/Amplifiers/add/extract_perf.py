import numpy as np
import os
import math

def analyze_amplifier_performance(vinp, vout, time, d0):
    vinp = np.array(vinp)  # Convert list to NumPy array
    vout = np.array(vout)
    time = np.array(time)

    # Function to extract step parameters from input signal
    def get_step_parameters(vinp, time):
        dv = np.diff(vinp)  # Calculate the difference between consecutive elements
        t0 = time[np.where(dv > 0)[0][0]]  # Detect rising edge time
        t1 = time[np.where(dv < 0)[0][0]]  # Detect falling edge time
        v0 = np.median(vinp[time < t0])  # Initial value before the step
        v1 = np.median(vinp[(time > t0) & (time < t1)])  # Value during the step
        return v0, v1, t0, t1

    # Extract step parameters from the input sequence
    v0, v1, t0, t1 = get_step_parameters(vinp, time)

    # Check amplifier stability before the step occurs
    pre_step_data = vout[time < t0]  # Data before the step
    delta0 = (pre_step_data - v0) / v0  # Calculate the percentage difference
    d0_settle = np.mean(np.abs(delta0))  # Calculate the average absolute difference
    stable = not np.any(np.abs(delta0) > d0)  # Check if amplifier is stable

    # Function to find the first index where the signal has settled
    def find_settling_time_index(delta, d0):
        for i in range(len(delta)):
            if np.all(np.abs(delta[i:]) < d0):  # Check if all subsequent values are within the threshold
                return i
        return None

    
    
    def get_slope_and_settling_time(vout, time, v0, v1, start_t, end_t, d0, mode):
        # Select data within the specified time range
        idx = (time >= start_t) & (time <= end_t)
        vout_segment = vout[idx]
        time_segment = time[idx]
    
        # Calculate the slope (Slew Rate, SR)
        target_value = v0 + (v1 - v0) / 2  # The midpoint between v0 and v1
        idx_target = np.where(vout_segment >= target_value)[0][0] if np.any(vout_segment >= target_value) else None
        if idx_target is None:
            SR = np.nan  # If no target value is reached, set SR to NaN
        else:
            SR = np.gradient(vout_segment, time_segment)[idx_target]  # Compute the gradient at the target point
    
        # Calculate settling time
        if mode == 'positive':
            delta = (vout_segment - v1) / v1  # Calculate the percentage deviation for the positive step
        else:
            delta = (vout_segment - v0) / v0  # Calculate the percentage deviation for the negative step
    
        # Find the first index where the signal settles (all subsequent deltas are below the threshold d0)
        idx_settle = find_settling_time_index(delta, d0)
        if idx_settle is None:
            settling_time = np.nan  # If the signal does not settle, return NaN
            d_settle = np.mean(np.abs(delta))  # Calculate the mean deviation
        else:
            settling_time = time_segment[idx_settle] - start_t  # Calculate the settling time
            d_settle = np.mean(np.abs(delta[idx_settle:]))  # Calculate the mean deviation after settling
    
        return SR, settling_time, d_settle

    
    
    # Calculate the slope and settling time for the rising edge
    SR_p, settling_time_p, d1_settle = get_slope_and_settling_time(vout, time, v0, v1, t0, t1, d0, 'positive')

    # Calculate the slope and settling time for the falling edge
    SR_n, settling_time_n, d2_settle = get_slope_and_settling_time(vout, time, v0, v1, t1, np.max(time), d0, 'negative')

    # Return all calculated metrics including stability, slope, and settling times for both edges
    return d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n

def extract_tran_data(path):
    time_points = []
    raw_data = []
    vin_data = []
    vout_data = []
    time_data = []
    data_section = False
    with open(path+'tran.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                if line.startswith('Values:'):
                    data_section = True
                    continue
                if data_section:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        time_points.append(int(parts[0]))
                        raw_data.append(float(parts[1]))
                    else:
                        raw_data.append(float(parts[0]))  
    
    if len(time_points) != len(raw_data)/3:
        print('Error in extracting transient data')
        return None, None
    for i in time_points:
        time_data.append(raw_data[3*i])
        vin_data.append(raw_data[3*i+2])
        vout_data.append(raw_data[3*i+1])
    
    return time_data, vin_data, vout_data


def get_tran_stable_meas(path):
    meas = {}
    d0 = 0.01  # Settling threshold
    time_data, vin_data, vout_data = extract_tran_data(path)  # Extract transient data
    if time_data is None:
        return None
    
    # Analyze amplifier performance and extract stability and slope metrics
    d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n = analyze_amplifier_performance(vin_data, vout_data, time_data, d0)
    #print(d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n)
    
    # Take the absolute values of the settling and slope values
    d0_settle = abs(d0_settle)
    d1_settle = abs(d1_settle)
    d2_settle = abs(d2_settle)
    SR_n = abs(SR_n)
    SR_p = abs(SR_p)
    settlingTime_p = abs(settling_time_p)
    settlingTime_n = abs(settling_time_n)

    # Handle NaN values for d0_settle
    if math.isnan(d0_settle):
        d0_settle = 10  # Assign a penalty value if d0_settle is NaN

    # Handle NaN values for d1_settle and d2_settle
    if math.isnan(d1_settle) or math.isnan(d2_settle):
        if math.isnan(d1_settle):
            d0_settle += 10  # Apply penalty if d1_settle is NaN
        if math.isnan(d2_settle):
            d0_settle += 10  # Apply penalty if d2_settle is NaN
        d_settle = d0_settle
    else:
        d_settle = max(d0_settle, d1_settle, d2_settle)  # Take the maximum settling value

    # Handle NaN values for SR (Slew Rate)
    if math.isnan(SR_p) or math.isnan(SR_n):
        SR = -d_settle  # Assign penalty if SR is not available
    else:
        SR = min(SR_p, SR_n)  # Take the minimum SR value

    # Handle NaN values for settling times
    if math.isnan(settlingTime_p) or math.isnan(settlingTime_n):
        settlingTime = d_settle  # Assign penalty if settling time is NaN
    else:
        settlingTime = max(settlingTime_p, settlingTime_n)  # Take the maximum settling time

    # Store the calculated metrics in the `meas` dictionary
    meas['d_settle'] = d_settle
    meas['SR'] = SR
    meas['settlingTime'] = settlingTime
    #print(meas)
    return meas



def extract_meas(path):
    meas = {}
    with open(path+'log.txt', 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.strip():
            if line.strip().split()[0] in spec_list:
                # print(line.strip().split(''))
                key, tmp,val = line.strip().split()[:3]
                # print(key, val)
                if key in meas.keys():
                    continue
                if val == 'failed':
                    continue
                else:
                    meas[key] = float(val)

    L_list = []
    W_list = []
    M_list = []
    R_list = []
    C_list = []
    with open(path+'param', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                if '_L_' in line:
                    L_list.append(float(line.strip().split('=')[-1]))
                elif '_W_' in line:
                    W_list.append(float(line.strip().split('=')[-1]))
                elif '_M_' in line:
                    M_list.append(float(line.strip().split('=')[-1]))
                elif 'RESISTOR' in line:
                    R_list.append(float(line.strip().split('=')[-1]))
                elif 'CAPACITOR' in line:
                    C_list.append(float(line.strip().split('=')[-1]))
                

        L_list = np.array(L_list)
        W_list = np.array(W_list)
        M_list = np.array(M_list)
        R_list = np.array(R_list)*1e-3*5
        C_list = np.array(C_list)*1e12*1085
        area = np.sum(L_list * W_list* M_list) + np.sum(R_list) + np.sum(C_list)
        

        meas['area'] = np.sqrt(area)

    with open(path+'log_tran.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                if line.strip().split()[0] in spec_list:
                    # print(line.strip().split(''))
                    key, tmp,val = line.strip().split()[:3]
                    # print(key, val)
                    if key in meas.keys():
                        continue
                    if val == 'failed':
                        continue
                    else:
                        meas[key] = float(val)

    for spec in spec_list:
        if spec not in meas.keys():
            if spec == 'dcgain':
                meas[spec] = -1000
            elif spec == 'gbp':
                meas[spec] = -1000
            elif spec == 'phase_in_deg':
                meas[spec] = 0
            elif spec == 'power':
                meas[spec] = 1000
            elif spec == 't_rise':
                meas[spec] = 1000
            elif spec == 't_fall':
                meas[spec] = 1000
            else:
                meas[spec] = np.nan
    meas_real = meas.copy()
    meas_real['power'] = meas['power']*1e-6
    meas_real['gbp'] = meas['gbp']*1e-2
    meas_real['t_rise'] = meas['t_rise']*1e-1
    meas_real['t_fall'] = meas['t_fall']*1e-1              
    #print(meas_real)
    return meas


if __name__ == '__main__':
    spec_list = ["cmrrdc", "dcgain", "gbp", "phase_in_rad", "phase_in_deg", "dcpsrp", "dcpsrn", "maxval","minval","avgval","ppavl","tc","ivdd25","power","vout25","vos25", "t_rise", "t_fall", "area"]
    meas = extract_meas('./')
    meas_tran = get_tran_stable_meas('./')
    meas.update(meas_tran)
    #print(meas)
    fom =  -abs(meas['gbp'])*10- meas['SR']*1e-3 -min(0,-100+meas['dcgain'])+ 50*abs(60-meas['phase_in_deg']) + meas['power']+ max(0,meas['area']-300) + meas['t_rise']/10 + max(0,-meas['t_fall']*10)+ 1e5*max(0,meas['settlingTime'])+meas['cmrrdc']*10 - max(abs(meas['dcpsrp']),abs(meas['dcpsrn']))*10
    foms = meas['gbp']*10 / (meas['power']*1e3)
    foml = meas['SR'] *10 / (meas['power']*1e3)
    #print('fom:', fom, 'foms:', foms, 'foml:', foml)
    with open("result.po", "w") as handler:
        handler.write("weighted_fom = {}".format(fom))
