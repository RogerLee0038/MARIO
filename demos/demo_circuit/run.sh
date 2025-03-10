#!/bin/bash
if [ -d "workspace" ]; then
  rm -rf "workspace"
fi
if [ -d "results" ]; then
  rm -rf "results"
fi
mkdir workspace
cp ${PO_SRC}/scripts/* workspace
cp ${PO_SRC}/auxi/* workspace
cp conf.toml workspace
cd workspace
python3 run_circuit.py > run.log
#python3 ABBO.py
mkdir ../results
#cp database* feedback_trades.pkl run.log summary.pkl ../results
cp database* summary.pkl *.pdf feedback.log run.log ../results
cd ..
cp conf.toml ${PO_SRC}/auxi/* results
rm -rf workspace
