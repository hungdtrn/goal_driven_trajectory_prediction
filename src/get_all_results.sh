#!/bin/bash

for predlen in {12,16,20,24,28}
    echo "observe 8, predict ${predlen}"
    do
        python -m src.get_results 
        --min_ped 2 --perspective 0 
        --use_goal 1 --pred_len ${predlen} 
        --model_name gtp
         --best 1 --test_all 1 ;
    done

