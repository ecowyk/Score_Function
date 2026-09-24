"""Deploy only the selected score branch; bind its sigma and frozen condition provenance."""

from score_function.model.score_branch import build_model
from score_function.utils.config import METHOD
from score_function.utils.train_utils import load_tensor


def load_selected(config, checkpoint, device):
    state = load_tensor(checkpoint)
    if (
        state.get("schema_version") != 1
        or state.get("method") != METHOD
        or "weight_kind" not in state
    ):
        raise ValueError(
            "Use score-function score/best.pt, not a resume or previous-method checkpoint"
        )
    if state["config"]["model"] != config["model"]:
        raise ValueError("Score architecture differs from checkpoint")
    if state["sigma_score"] != config["training"]["sigma"]:
        raise ValueError("Inference sigma must equal this branch's training sigma")
    model = build_model(state["config"], device)
    model.load_state_dict(state["score_branch"], strict=True)
    model.eval().requires_grad_(False)
    return model, state
