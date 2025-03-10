def get_num(str):
    str = str.lower()
    if str.endswith('f'):
        num = float(str.split('f')[0]+'e-15')
    elif str.endswith('p'):
        num = float(str.split('p')[0]+'e-12')
    elif str.endswith('n'):
        num = float(str.split('n')[0]+'e-9')
    elif str.endswith('u'):
        num = float(str.split('u')[0]+'e-6')
    elif str.endswith('m'):
        num = float(str.split('m')[0]+'e-3')
    elif str.endswith('k'):
        num = float(str.split('k')[0]+'e3')
    elif str.endswith('meg'):
        num = float(str.split('meg')[0]+'e6')
    else:
        num = float(str)
    return num

def get_lb(name, num, scale=0.2):
    if '_M_' in name:
        limit = 1
    elif 'RESISTOR' in name:
        limit = 1e3
    elif 'CAPACITOR' in name:
        limit = 1e-15
    elif 'CURRENT' in name:
        limit = 0
    else:
        limit = 0.5
    lb = max(num*scale, limit)
    return lb

def get_ub(name, num, scale=5):
    if '_M_' in name:
        limit = 900
    elif 'RESISTOR' in name:
        limit = 900e3
    elif 'CAPACITOR' in name:
        limit = 500e-12
    elif 'CURRENT' in name:
        limit = 100e-6
    else:
        limit = 5
    ub = min(num*scale, limit)
    return ub

with open('circuit/param','r') as handler:
    with open('DX.txt', 'w') as wf:
        for line in handler.readlines():
            param_list = line.strip().split('=')
            param_name = param_list[0].split()[1]
            param_num = get_num(param_list[1])
            param_lb = get_lb(param_name, param_num)
            param_ub = get_ub(param_name, param_num)
            if '_M_' in param_name:
                param_step = 1
            elif 'RESISTOR' in param_name:
                param_step = 1e2
            elif 'CAPACITOR' in param_name:
                param_step = 1e-15
            elif 'CURRENT' in param_name:
                param_step = 1e-7
            else:
                param_step = 1e-1
            dx=(param_name,
                float("{:.6g}".format(param_lb)), 
                float("{:.6g}".format(param_ub)), 
                param_step, 
                float("{:.6g}".format(param_num)),
                'NO')
            print(dx, end=',\n', file=wf)
