import numpy as np
import pandas as pd

from omegaconf import OmegaConf


def softly_vote_logits(
    config_path: str,
) -> None:
    config = OmegaConf.load(config_path)
    logit_files = config.logit_files
    weights = config.weights
    submission_file_path = config.submission_file_path
    target_column_name = config.target_column_name
    voting_file_path = config.voting_file_path

    if len(logit_files) != len(weights):
        raise ValueError(
            f"logit_file numbers({len(logit_files)}) does not match with weight numbers({len(weights)})"
        )
    if not np.isclose(sum(weights), 1):
        raise ValueError(f"summation of weights({sum(weights)}) is not equal to 1")

    weighted_logits = None
    for logit_file, weight in zip(logit_files, weights):
        try:
            logit = np.load(logit_file)
        except:
            raise FileNotFoundError(f"logit file {logit_file} dose noe exist")
        if weighted_logits is None:
            weighted_logits = logit * weight
        else:
            weighted_logits += logit * weight

    ensemble_predictions = np.argmax(
        weighted_logits,
        axis=1,
    )
    submission_df = pd.read_csv(submission_file_path)
    submission_df[target_column_name] = ensemble_predictions
    submission_df.to_csv(
        voting_file_path,
        index=False,
    )


if __name__ == "__main__":
    CONFIG_PATH = "./voting_config.yaml"
    softly_vote_logits(config_path=CONFIG_PATH)
