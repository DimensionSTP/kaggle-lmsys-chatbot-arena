import dotenv

dotenv.load_dotenv(
    override=True,
)

from typing import List
import os

import pandas as pd

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def preprocess_dataset(
    config: DictConfig,
) -> None:
    df = pd.read_csv(f"{config.connected_dir}/data/{config.mode}.csv")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{config.custom_data_encoder_path}/{config.pretrained_model_name}",
        use_fast=True,
    )

    def generate_prompt(
        datas: List[str],
    ) -> str:
        default_system_prompt = """
You are tasked with predicting which response people would prefer. 
Read the given prompt and the two responses below, then choose which response you think people will like more.
If you think people will prefer model a's response, select 0. 
If you think people will prefer model b's response, select 1. 
If you think it's a tie, select 2. 
You must answer only with 0, 1, or 2.
"""
        prompt = f"""### Instruction:
{default_system_prompt} 

### Input:

**Prompt**:
{datas[0].strip()}

**Answer of model a**:
{datas[1].strip()}

**Answer of model b**:
{datas[2].strip()}

### Response:
Choose between 0, 1, or 2: """.strip()
        return prompt

    df["total_prompt"] = df.apply(
        lambda row: generate_prompt(
            [
                row[config.data_column_names[0]],
                row[config.data_column_names[1]],
                row[config.data_column_names[2]],
            ]
        ),
        axis=1,
    )
    for data_column_name in config.data_column_names:
        df[data_column_name] = df[data_column_name].apply(lambda x: x.strip())

    def cut_prompt_to_length(
        prompt: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> str:
        tokens = tokenizer.tokenize(prompt)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        cut_prompt = tokenizer.convert_tokens_to_string(tokens)
        return cut_prompt

    df[config.prompt_column_name] = df["total_prompt"].apply(
        lambda x: cut_prompt_to_length(
            prompt=x,
            tokenizer=tokenizer,
            max_length=config.data_max_length,
        )
    )
    if not os.path.exists(
        f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}"
    ):
        os.makedirs(
            f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}",
            exist_ok=True,
        )
    df.to_csv(
        f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}/{config.mode}.csv",
        index=False,
    )


if __name__ == "__main__":
    preprocess_dataset()
