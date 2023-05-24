# Data util classes copied from RL4LMs repo: https://github.com/allenai/RL4LMs/blob/main/rl4lms/data_pools/custom_text_generation_pools.py
import random
from abc import abstractclassmethod
from dataclasses import dataclass
from typing import Any, List, Dict

@dataclass(init=True)
class Sample:
    id: str
    prompt_or_input_text: str
    references: List[str]
    meta_data: Dict[str, Any] = None


class TextGenPool:
    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @abstractclassmethod
    def prepare(cls, **args) -> 'TextGenPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List['TextGenPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix]))
            start_ix = end_ix
        return pools

from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import os
from urllib.request import urlretrieve
from pathlib import Path
import pandas
from collections import defaultdict
import zipfile
import json

class CommonGen(TextGenPool):
    @classmethod
    def prepare(cls, split: str,
                concept_separator_token: str = " ",
                concept_end_token=" ",
                prefix: str = "summarize: ") -> 'TextGenPool':
        ds = load_dataset("gem", "common_gen")
        samples = []
        split_id = CommonGen.gen_split_name(split)
        for ix, item in enumerate(ds[split_id]):
            concepts = concept_separator_token.join(item["concepts"])
            concepts = prefix + concepts
            concepts += concept_end_token
            if item["target"] == "":
                # just to avoid breaking of metric computation
                item["target"] = "empty reference"
            targets = [item["target"]]
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=concepts,
                            references=targets,
                            meta_data={
                                "concepts": item["concepts"]
                            }
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "validation"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name


class Xsum(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "TL;DR:"):
        dataset = load_dataset("gem", "xsum")
        dataset_split = dataset[split]
        samples = []
        for ix, item in enumerate(dataset_split):
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=item["document"] +
                            prompt_suffix,
                            references=[item["target"]]
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance


class CNNDailyMail(TextGenPool):
    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                truncate_article: int = None,
                max_size: int = None):
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        dataset_split = CommonGen.gen_split_name(split)
        samples = []
        for ix, item in tqdm(enumerate(dataset[dataset_split]),
                             desc="Tokenizing dataset",
                             total=len(dataset[dataset_split])):

            if truncate_article is not None:
                tokens = word_tokenize(item["article"])
                tokens = tokens[:truncate_article]
                item["article"] = " ".join(tokens)

            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt_prefix +
                            item["article"] + prompt_suffix,
                            references=[item["highlights"]]
                            )
            samples.append(sample)

            if max_size is not None and ix == (max_size-1):
                break

        pool_instance = cls(samples)
        return pool_instance


class IMDBForSeq2Seq(TextGenPool):
    """
    IMDB Dataset in seq2seq format to train supervised generator
    """
    @classmethod
    def prepare(cls, split: str, positive_ratio: int = 1.0):
        # dataset = load_dataset("imdb")
        # NOTE: there is a bug in huggingface dataset regarding imdb dataset
        # Ref: https://github.com/huggingface/datasets/issues/4550#issuecomment-1164407965
        # dataset = load_dataset("imdb", revision="tmp-fix-imdb", ignore_verifications=True)
        dataset = load_dataset("imdb", ignore_verifications=True)
        if split in ["train", "val"]:
            dataset_split = dataset["train"].shuffle()
            train_ratio = 0.8
            train_index = int(len(dataset_split) * train_ratio)
            dataset_split = dataset_split[:train_index] if split == "train" else dataset_split[train_index:]
        else:
            # limit test to 5000
            dataset_split = dataset[split].shuffle()
            dataset_split = dataset_split[:5000]

        samples = []
        for ix, (text, label) in enumerate(zip(dataset_split["text"], dataset_split["label"])):

            # here we consider 50% of tokens as prompt and rest as references
            tokenized_text = text.split(" ")
            text_split_index = int(len(tokenized_text) * 0.5)
            prompt_text = " ".join(tokenized_text[:text_split_index])
            ref_text = " ".join(tokenized_text[text_split_index:])

            # add only positive examples for train set
            if split == "train" and label == 1 or split != "train":
                sample = Sample(id=f"{split}_{ix}",
                                prompt_or_input_text=prompt_text,
                                references=[ref_text],
                                meta_data={
                                    "reference": text
                                }
                                )
                samples.append(sample)

        # truncate train split
        if split == "train":
            samples = samples[:int(len(samples) * positive_ratio)]

        pool_instance = cls(samples)
        return pool_instance


def download_file_using_url(url: str, dest_path: str):
    urlretrieve(url, dest_path)

class WMT(TextGenPool):
    @classmethod
    def get_dataset(cls, wmt_id: str, source_language: str, target_language: str, split: str):
        try:
            language_pair = f"{source_language}-{target_language}"
            dataset = load_dataset(f"{wmt_id}", language_pair)
        except:
            language_pair = f"{target_language}-{source_language}"
            dataset = load_dataset(f"{wmt_id}", language_pair)
        dataset = dataset[split]
        return dataset

    @classmethod
    def prepare(cls,
                wmt_id: str,
                split: str,
                source_language: str,
                target_language: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = WMT.get_dataset(
            wmt_id, source_language, target_language, dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"][source_language] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"][target_language]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


class WMT14PreprocessedEnDe(TextGenPool):
    @classmethod
    def get_dataset(cls, split: str):
        dataset = load_dataset("stas/wmt14-en-de-pre-processed")[split]
        return dataset

    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = WMT14PreprocessedEnDe.get_dataset(dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"]["en"] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"]["de"]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


class WMT16NewsOnlyDatasetEnDe(TextGenPool):
    @classmethod
    def get_dataset(cls, split: str):

        if split == "train":
            dataset = load_dataset("news_commentary", "de-en")[split]
            return dataset
        else:
            dataset = load_dataset("wmt16", "de-en")[split]
        return dataset

    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = WMT16NewsOnlyDatasetEnDe.get_dataset(dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"]["en"] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"]["de"]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


class IWSLT2017EnDe(TextGenPool):
    @classmethod
    def get_dataset(cls, split: str):
        dataset = load_dataset(
            "iwslt2017", "iwslt2017-de-en", ignore_verifications=True)
        return dataset[split]

    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = IWSLT2017EnDe.get_dataset(dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"]["en"] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"]["de"]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance

class DailyDialog(TextGenPool):
    EOU_TOKEN = "<EOU>"
    @classmethod
    def prepare(cls, split: str, context_size: int):
        split = CommonGen.gen_split_name(split)
        dataset = load_dataset("daily_dialog", split=split)
        samples = []
        utterance_id = 0
        for item in dataset:
            contexts = []
            for utterance, emotion, intent in zip(item["dialog"],
                                                  item["emotion"],
                                                  item["act"]):
                if len(contexts) >= context_size:
                    context = DailyDialog.EOU_TOKEN.join(contexts[-context_size:]) 
                    context += " " + DailyDialog.EOU_TOKEN
                    target = utterance + DailyDialog.EOU_TOKEN
                    sample = Sample(id=utterance_id, 
                                    prompt_or_input_text=context, 
                                    references=[target],
                                    meta_data={
                                        "emotion": [emotion],
                                        "intent": [intent]
                                    })
                    samples.append(sample)
                contexts.append(utterance)
                utterance_id += 1

        dp_instance = cls(samples)
        return dp_instance
