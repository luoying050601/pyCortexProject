# shellcheck disable=SC2006
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"
# shellcheck disable=SC2034
#model_type='brainbert'
#cherry
#nohup python3 /home/ying/project/pycortexProj/com/pycortex/train_musicRidgeRegression.py \
# laplace
nohup python3.7 /Storage/ying/pyCortexProj/com/pycortex/train_musicRidgeRegression.py \
--type 'rr' \
--arpha 0.1 \
>>"music_rr_${cur_date}".out 2>&1 &

#nohup python3 ridge.py \
#--type 'bert' \
#--arpha 0.1 \
#>>"ridge_regression_bert_${cur_date}".out 2>&1 &
