from sl import config
from sl.utils import fn_utils


def get_repo_name(model_name: str) -> str:
    assert config.HF_USER_ID != ""
    return f"{config.HF_USER_ID}/{model_name}"


# runpod has flaky db connections...
@fn_utils.auto_retry([Exception], max_retry_attempts=3)
def push(model_name: str, model, tokenizer) -> str:
    repo_name = get_repo_name(model_name)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    return repo_name
