# shellcheck disable=SC2006
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"
# shellcheck disable=SC2034
#model_type='brainbert'
#cherry
# laplace
nohup /usr/bin/python3 -u /Storage2/ying/pyCortexProj/ridgeRegression/text_embedding/littlePrince/create_word_embedding_prince.py \
>>"dcreate_word_embedding_brainlm_${cur_date}".out 2>&1 &
