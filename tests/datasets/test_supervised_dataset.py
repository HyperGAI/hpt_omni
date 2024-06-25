import unittest
from transformers import AutoTokenizer
from llava.data.dataset import LazySupervisedDataset
from llava.train.args import DataArguments


class TestSupervisedDataset(unittest.TestCase):

    def test(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "/export/share/models/Meta-Llama-3-8B-Instruct/",
            model_max_length=10000,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        data_args = DataArguments(
            data_path="/export/share/dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
            image_folder="/export/share/dataset/LLaVA-Pretrain/images",
            data_mixture="blip_laion_cc_sbu_558k"
        )
        dataset = LazySupervisedDataset(
            data_path="/export/share/dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
            image_folder="/export/share/dataset/LLaVA-Pretrain/images",
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=None
        )
        for i in range(10):
            x = dataset.__getitem__(i)
            print(x)
            input("next?")


if __name__ == "__main__":
    unittest.main()
