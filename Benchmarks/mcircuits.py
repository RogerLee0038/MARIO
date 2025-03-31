#from mcircuitlib.ACCIA_38dim.accia import ACCIA 
#from mcircuitlib.chargepump.chargepump import ChargePump
#from mcircuitlib.gainBoost.gainboost import GainBoost
#from mcircuitlib.OTA_59dim.ota import OTA
#from mcircuitlib.two_stage_opamp.two_stage_opamp import TwoStageOpamp
#from mcircuitlib.VCO.vco import VCO
#from mcircuitlib.PLLVCO.pll_vco import PLLVCO
#from mcircuitlib.classE.classE import classE
#from mcircuitlib.LNA.lna import LNA
#from mcircuitlib.three_stage_opamp.three_stage_opamp import ThreeStageOpamp
#from mcircuitlib.PA.pa import PA
from mAnalogGym.Amplifiers.Alfio_RAFFC_Pin_3.Alfio_RAFFC_Pin_3 import Alfio_RAFFC
from mAnalogGym.Amplifiers.Fan_SMC_Pin_3.Fan_SMC_Pin_3 import Fan_SMC
from mAnalogGym.Amplifiers.HoiLee_AFFC_Pin_3.HoiLee_AFFC_Pin_3 import HoiLee_AFFC
from mAnalogGym.Amplifiers.Leung_DFCFC1_Pin_3.Leung_DFCFC1_Pin_3 import Leung_DFCFC1
from mAnalogGym.Amplifiers.Leung_DFCFC2_Pin_3.Leung_DFCFC2_Pin_3 import Leung_DFCFC2
from mAnalogGym.Amplifiers.Leung_NMCF_Pin_3.Leung_NMCF_Pin_3 import Leung_NMCF
from mAnalogGym.Amplifiers.Leung_NMCNR_Pin_3.Leung_NMCNR_Pin_3 import Leung_NMCNR
from mAnalogGym.Amplifiers.Peng_ACBC_Pin_3.Peng_ACBC_Pin_3 import Peng_ACBC
from mAnalogGym.Amplifiers.Peng_IAC_Pin_3.Peng_IAC_Pin_3 import Peng_IAC
from mAnalogGym.Amplifiers.Qu2017_AZC_Pin_3.Qu2017_AZC_Pin_3 import Qu2017_AZC
from mAnalogGym.Amplifiers.Ramos_PFC_Pin_3.Ramos_PFC_Pin_3 import Ramos_PFC
from mAnalogGym.Amplifiers.Sau_CFCC_Pin_3.Sau_CFCC_Pin_3 import Sau_CFCC
from mAnalogGym.Amplifiers.Song_DACFC_Pin_3.Song_DACFC_Pin_3 import Song_DACFC
from mAnalogGym.Amplifiers.Yan_AZ_Pin_3.Yan_AZ_Pin_3 import Yan_AZ

#maccia = ACCIA()
#mchargepump = ChargePump()
#mgainboost = GainBoost()
#mota = OTA()
#mtwo_stage_opamp = TwoStageOpamp()
#mvco = VCO()
#mpll_vco = PLLVCO()
#mclass_e = classE()
#mlna = LNA()
#mthree_stage_opamp = ThreeStageOpamp()
#mpa = PA()
mAlfio_RAFFC = Alfio_RAFFC()
mFan_SMC= Fan_SMC()
mHoiLee_AFFC= HoiLee_AFFC()
mLeung_DFCFC1= Leung_DFCFC1()
mLeung_DFCFC2= Leung_DFCFC2()
mLeung_NMCF= Leung_NMCF()
mLeung_NMCNR= Leung_NMCNR()
mPeng_ACBC= Peng_ACBC()
mPeng_IAC= Peng_IAC()
mQu2017_AZC= Qu2017_AZC()
mRamos_PFC= Ramos_PFC()
mSau_CFCC= Sau_CFCC()
mSong_DACFC= Song_DACFC()
mYan_AZ= Yan_AZ()

if __name__ == '__main__':
    #print("accia", maccia.in_dim)
    #print("chargepump", mchargepump.in_dim)
    #print("gainboost", mgainboost.in_dim)
    #print("ota", mota.in_dim)
    #print("two_stage_opamp", mtwo_stage_opamp.in_dim)
    #print("vco", mvco.in_dim)
    #print("pll_vco", mpll_vco.in_dim)
    #print("class_e", mclass_e.in_dim)
    #print("lna", mlna.in_dim)
    #print("three_stage_opamp", mthree_stage_opamp.in_dim)
    #print("pa", mpa.in_dim) 
    print("Alfio_RAFFC", mAlfio_RAFFC.in_dim) 
    print("Fan_SMC", mFan_SMC.in_dim) 
    print("HoiLee_AFFC", mHoiLee_AFFC.in_dim) 
    print("Leung_DFCFC1", mLeung_DFCFC1.in_dim) 
    print("Leung_DFCFC2", mLeung_DFCFC2.in_dim) 
    print("Leung_NMCF", mLeung_NMCF.in_dim) 
    print("Leung_NMCNR", mLeung_NMCNR.in_dim) 
    print("Peng_ACBC", mPeng_ACBC.in_dim) 
    print("Peng_IAC", mPeng_IAC.in_dim) 
    print("Qu2017_AZC", mQu2017_AZC.in_dim) 
    print("Ramos_PFC", mRamos_PFC.in_dim) 
    print("Sau_CFCC", mSau_CFCC.in_dim) 
    print("Song_DACFC", mSong_DACFC.in_dim) 
    print("Yan_AZ", mYan_AZ.in_dim) 
