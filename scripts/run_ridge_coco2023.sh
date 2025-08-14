# shellcheck disable=SC2006
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"
# shellcheck disable=SC2034
#model_type='brainbert'
#cherry
#nohup python3 /home/ying/project/pycortexProj/com/pycortex/train_musicRidgeRegression.py \
# laplace
nohup /usr/local/bin/python3.7 /Storage/ying/pyCortexProj/ridgeRegression/ridge.py \
>> "ridgeRegression_${cur_date}".out 2>&1 &