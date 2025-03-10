#!/bin/bash
ngspice  -o log.txt -b TB_Amplifier_ACDC.cir 1>acdc.out 2>&1
ngspice  -o log_tran.txt -b TB_Amplifier_Tran.cir 1>tran.out 2>&1
