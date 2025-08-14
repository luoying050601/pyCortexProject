# shellcheck disable=SC2006
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"
# shellcheck disable=SC2034
# thales
nohup /usr/bin/python3.6 -u /Storage/ying/pyCortexProj/com/pycortex/create_experimental_random_file.py \
--close_single 10 \
--close_chord 15 \
--close_eye 8 \
--open_eyes 15 \
--diffcult False \
--Subject_id SM0001 \
--current_diff_size 4 \
>>"experimental_random_${cur_date}".out 2>&1 &
