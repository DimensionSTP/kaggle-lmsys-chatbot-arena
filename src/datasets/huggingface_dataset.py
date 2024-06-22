from typing import Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class KaggleChatbotArenaDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        is_preprocessed: bool,
        data_column_names: List[str],
        prompt_column_name: str,
        target_column_names: List[str],
        label_column_name: str,
        num_devices: int,
        batch_size: int,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        data_max_length: int,
        target_max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.is_preprocessed = is_preprocessed
        self.data_column_names = data_column_names
        self.prompt_column_name = prompt_column_name
        self.target_column_names = target_column_names
        self.label_column_name = label_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.pretrained_model_name = pretrained_model_name
        if self.is_preprocessed:
            data_encoder_path = (
                f"{custom_data_encoder_path}/{self.pretrained_model_name}"
            )
        else:
            data_encoder_path = self.pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            data_encoder_path,
            use_fast=True,
        )
        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.data_max_length = data_max_length
        self.target_max_length = target_max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        if self.is_preprocessed:
            prompt = self.datas[idx] + str(self.labels[idx])
        else:
            datas = []
            for data_column_idx in range(len(self.datas)):
                datas.append(self.datas[data_column_idx][idx])
            prompt = self.generate_prompt(
                datas=datas,
                label=self.labels[idx],
            )
        encoded = self.encode_text(
            data=prompt,
        )
        encoded["labels"] = torch.tensor(
            [self.labels[idx]],
            dtype=torch.long,
        ).squeeze(0)
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        return {
            "encoded": encoded,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/train.csv"
            else:
                csv_path = f"{self.data_path}/train.csv"
            data = pd.read_csv(csv_path)
            data = data[(data[self.target_column_names].sum(axis=1) == 1)]
            data = data.fillna("_")
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
                stratify=data[self.target_column_names],
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/{self.split}.csv"
            else:
                csv_path = f"{self.data_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
            data = data[(data[self.target_column_names].sum(axis=1) == 1)]
            data = data.fillna("_")
        elif self.split == "predict":
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/test.csv"
            else:
                csv_path = f"{self.data_path}/test.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")
        if self.is_preprocessed:
            datas = data[self.prompt_column_name].tolist()
        else:
            datas = []
            for data_column_name in self.data_column_names:
                datas.append(data[data_column_name].apply(lambda x: x.strip()).tolist())
        data[self.label_column_name] = data.apply(
            self.convert_labels,
            axis=1,
        )
        labels = data[self.label_column_name].tolist()
        return {
            "datas": datas,
            "labels": labels,
        }

    def convert_labels(
        self,
        row: pd.Series,
    ) -> int:
        if row[self.target_column_names[0]] == 1:
            return 0
        elif row[self.target_column_names[1]] == 1:
            return 1
        elif row[self.target_column_names[2]] == 1:
            return 2
        else:
            return 2

    def encode_text(
        self,
        data: str,
    ) -> Dict[str, torch.Tensor]:
        if self.split == "predict":
            max_length = self.data_max_length
        else:
            max_length = self.data_max_length + self.target_max_length
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def generate_prompt(
        self,
        datas: List[str],
        label: str,
    ) -> str:
        default_system_prompt = """
You are tasked with predicting which response people would prefer. 
Read the given prompt and the two responses below, then choose which response you think people will like more.
If you think people will prefer model a's response, select 0. 
If you think people will prefer model b's response, select 1. 
If you think it's a tie, select 2. 
You must answer only with 0, 1, or 2.
"""
        if self.split == "predict":
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
        else:
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
Choose between 0, 1, or 2: {label} """.strip()
        return prompt
