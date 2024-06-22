import dotenv

dotenv.load_dotenv(
    override=True,
)

import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def add_empty_column(
    config: DictConfig,
) -> None:
    df = pd.read_csv(f"{config.connected_dir}/data/{config.submission_file_name}.csv")
    for target_column_name in config.target_column_names:
        df[target_column_name] = 0
    df.to_csv(
        f"{config.connected_dir}/data/{config.submission_file_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    add_empty_column()
