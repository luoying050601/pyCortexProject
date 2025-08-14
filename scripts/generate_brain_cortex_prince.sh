# shellcheck disable=SC2006
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"
# shellcheck disable=SC2034
#model_type='brainbert'
#cherry
#nohup python3 /home/ying/project/pycortexProj/com/pycortex/train_musicRidgeRegression.py \
# laplace
nohup /usr/bin/python3 -u /Storage2/ying/pyCortexProj/ridgeRegression/plot_rr_result_flatmap.py \
>>"generate_prince_brain_cortex_${cur_date}".out 2>&1 &
