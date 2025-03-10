import os

cirNames = os.listdir('/export/home/liwangzhen/TestSpace/AnalogGymTest/AnalogGym/Amplifier/spice_netlist')
print(cirNames)
for cirName in cirNames:
    os.system('mkdir -p {}/circuit'.format(cirName))
    os.system('cp /export/home/liwangzhen/TestSpace/AnalogGymTest/AnalogGym/Amplifier/design_variables/{0} {0}/circuit/param'.format(cirName))
    os.system('cp /export/home/liwangzhen/TestSpace/AnalogGymTest/AnalogGym/Amplifier/spice_netlist/{0} {0}/circuit'.format(cirName))
    os.system('cp /export/home/liwangzhen/TestSpace/AnalogGymTest/AnalogGym/Amplifier/amp_spice_testbench/* {}/circuit'.format(cirName))
    with open('{}/circuit/TB_Amplifier_ACDC.cir'.format(cirName), 'r') as tbr_acdc:
        content_acdc = tbr_acdc.read()
        new_acdc = content_acdc.replace('HoiLee_AFFC_Pin_3', '{}'.format(cirName))
    with open('{}/circuit/TB_Amplifier_ACDC.cir'.format(cirName), 'w') as tbw_acdc:
        tbw_acdc.write(new_acdc)
    with open('{}/circuit/TB_Amplifier_Tran.cir'.format(cirName), 'r') as tbr_tran:
        content_tran = tbr_tran.read()
        new_tran = content_tran.replace('HoiLee_AFFC_Pin_3', '{}'.format(cirName))
    with open('{}/circuit/TB_Amplifier_Tran.cir'.format(cirName), 'w') as tbw_tran:
        tbw_tran.write(new_tran)
