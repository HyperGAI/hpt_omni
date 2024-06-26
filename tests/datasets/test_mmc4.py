import os
import unittest
from transformers import AutoTokenizer
from llava import conversation as conversation_lib
from llava.data.dataset import LazyMMC4Dataset
from llava.train.args import DataArguments, TrainingArguments
from llava.model.multimodal_encoder.intern_encoder import InternVisionPreprocessor


class TestMMC4(unittest.TestCase):

    def test(self):
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
        tokenizer = AutoTokenizer.from_pretrained(
            "/export/share/models/Meta-Llama-3-8B-Instruct/",
            model_max_length=10000,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        data_args = DataArguments(
            data_path="/export/share/dataset/MMC4/pkl_core",
            data_mixture="mmc4core"
        )
        data_args.image_processor = InternVisionPreprocessor()
        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = False

        training_args = TrainingArguments(output_dir=".")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        dataset = LazyMMC4Dataset(
            data_path="/export/share/dataset/MMC4/pkl_core",
            image_folder=None,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args
        )
        for i in range(10):
            x = dataset.__getitem__(i)
            print(x)
            input("next?")


if __name__ == "__main__":
    unittest.main()
