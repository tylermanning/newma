#!/usr/bin/env bash
which python
for algo in 'newmaRFF' 'ScanB'
do
    python test_B_runningtime.py ${algo}
done
