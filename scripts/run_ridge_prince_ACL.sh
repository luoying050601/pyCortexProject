# shellcheck disable=SC2006
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"
# shellcheck disable=SC2034
#model_type='brainbert'
#cherry
#nohup python3 /home/ying/project/pycortexProj/com/pycortex/train_musicRidgeRegression.py \
# laplace　Storage2
#nohup /usr/bin/python3 -u /Storage2/ying/pyCortexProj/ridgeRegression/ridge_prince_Storage2.py \
#>> "ridgeRegression_prince_${cur_date}".out 2>&1 &
nohup /usr/bin/python3 -u /home/ying/project/pyCortexProj/ridgeRegression/ridge_prince_home.py \
>> "ridgeRegression_prince_${cur_date}".out 2>&1 &