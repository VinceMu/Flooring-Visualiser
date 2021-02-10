import os, json


def provide_model_params(model_dir):
    params_path = f"{model_dir}/params.json"
    if os.path.isfile(params_path):
        with open(params_path) as params_file:
            params_dict = json.load(params_file)
        return (params_dict["encoder"], params_dict["decoder"])
    else:
        return ({"fc_dim": 2048}, {"fc_dim": 2048})
