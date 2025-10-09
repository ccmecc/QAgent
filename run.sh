reward_rule="EM"
reference_path="XXX"
model_path="XXX"
job_name="XXX"

ENVS="reward_rule=${reward_rule},log_name=${job_name},model_path=${model_path},reference_path=${reference_path}"
if [ -n "${check_point_dir}" ]; then
  ENVS="${ENVS},check_point_dir=${check_point_dir}"
fi

export ${ENVS}


####### TO Be Complete #########