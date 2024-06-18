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
def scale_target_column(
    config: DictConfig,
) -> None:
    df = pd.read_csv(f"{config.connected_dir}/data/{config.split.train}.csv")
    df.to_csv(
        f"{config.connected_dir}/data/original_{config.split.train}.csv",
        index=False,
    )
    df[config.target_column_name] = df[config.target_column_name] - 1
    df.to_csv(
        f"{config.connected_dir}/data/{config.split.train}.csv",
        index=False,
    )


if __name__ == "__main__":
    scale_target_column()
