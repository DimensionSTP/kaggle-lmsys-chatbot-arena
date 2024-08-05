import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="voting.yaml",
)
def softly_vote_probabilities(
    config: DictConfig,
) -> None:
    connected_dir = config.connected_dir
    voted_probability = config.voted_probability
    submission_file = config.submission_file
    target_column_names = config.target_column_names
    voted_file = config.voted_file
    votings = config.votings

    weights = list(votings.values())
    if not np.isclose(sum(weights), 1):
        raise ValueError(f"summation of weights({sum(weights)}) is not equal to 1")

    weighted_probabilities = None
    for probability_file, weight in votings.items():
        try:
            probability = np.load(
                f"{connected_dir}/probabilities/{probability_file}.npy"
            )
        except:
            raise FileNotFoundError(
                f"probability file {probability_file} does not exist"
            )
        if weighted_probabilities is None:
            weighted_probabilities = probability * weight
        else:
            weighted_probabilities += probability * weight

    submission_df = pd.read_csv(submission_file)
    np.save(
        voted_probability,
        weighted_probabilities,
    )
    for i, target_column_name in enumerate(target_column_names):
        submission_df[target_column_name] = weighted_probabilities[:, i]
    submission_df.to_csv(
        voted_file,
        index=False,
    )


if __name__ == "__main__":
    softly_vote_probabilities()
