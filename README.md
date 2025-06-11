## Overview
This is the code-release for the MARIO optimization framework from "MARIO: A Superadditive Multi-Algorithm Interworking Optimization Framework for Analog Circuit Sizing" appearing in DAC 2025.

## Dependencies
See requirments.txt

## Involved Algorithms 
See scripts or scripts_circuit

## Function and Circuit Benchmarks
See Benchmarks

## Demos: How to Run Experiments
In Demos, demo for test functions provided by Nevergrad, demo_go for Global Optimization (GO) functions provided by Scipy, demo_goc for GO functions with scalable dimensionality collected by us, demo_circuit for analog-circuit testcases provided by AnalogGym.  
E.g.
1. `source env.bashrc`
2. `cd Demos/demo_goc`
3. Edit confxp.toml
4. `./runxp.sh`
5. raw data in directory 'alldatas', optimization records in the csv file

## Results and Post-Processing
See Results, according to the raw data and csv file, the optimization curves and breakdown can be plotted.

## Citing us