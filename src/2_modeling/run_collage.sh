#!/bin/bash

for pred_task in 0 1 2; do
    for corr_thresh in 0.99 0.95 0.9 0.8 0.7 0.6 0.5; do
        python collage_lasso_nltocv.py --pred_task $pred_task --corr_thresh $corr_thresh
    done
done
