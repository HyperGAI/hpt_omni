from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)


DATASETS = {}

import warnings


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})



def register_datasets_mixtures():
    # S1
    llava_1_5_mm_align = Dataset(
        dataset_name='llava_1_5_align',
        dataset_type='torch',
        data_path='./playground/data/LLaVA-Pretrain/LLaVA-CC3M-Pretrain-595K.json',
        image_path='./playground/data/LLaVA-Pretrain/images'
    )
    add_dataset(llava_1_5_mm_align)
    
    blip_laion_cc_sbu_558k = Dataset(
        dataset_name='blip_laion_cc_sbu_558k',
        dataset_type='torch',
        data_path='/export/share/dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json',
        image_path='/export/share/dataset/LLaVA-Pretrain/images'
    )
    add_dataset(blip_laion_cc_sbu_558k)
    
    # S2
    coyo_25m = Dataset(
        dataset_name='coyo',
        dataset_type='coyo',
        data_path='/export/share/dataset/COYO/coyo-700m/pkl02-split'
        # data_path='/export/share/dataset/llava1.6/interleaved_data/10M/coyo'
        )
    add_dataset(coyo_25m)

    mmc4core = Dataset(
        dataset_name='mmc4core',
        dataset_type='mmc4',
        # data_path='/export/share/dataset/llava1.6/interleaved_data/10M/mmc4',
        data_path='/export/share/dataset/MMC4/pkl_core'
        )
    add_dataset(mmc4core)

    sharegpt4v_pretrain = Dataset(
        dataset_name="sharegpt4v_pretrain",
        dataset_type="torch",
        data_path="/export/share/dataset/ShareGPT4V/share-captioner_coco_lcs_sam_1246k_1107_checked.json",
        image_path="/export/share/dataset/ShareGPT4V/data",
    )
    add_dataset(sharegpt4v_pretrain)
    
    struct_aware_parse_3m = Dataset(
        dataset_name="struct_aware_parse_3m",
        dataset_type="torch",
        data_path='/export/share/yucheng/playground/hpt2.0/ocr_related/struct_aware_parse_3m_checked.json',
        image_path='/export/share/dataset/llava1.6',
    )
    add_dataset(struct_aware_parse_3m)
    
    
    multi_grained_text_localization_1m = Dataset(
        dataset_name="multi_grained_text_localization_1m",
        dataset_type="torch",
        data_path='/export/share/yucheng/playground/hpt2.0/ocr_related/multi_grained_text_localization_1m_checked.json',
        image_path='/export/share/dataset/llava1.6',
    )
    add_dataset(multi_grained_text_localization_1m)
    
    
    # S3
    hpt_v41 = Dataset(
        dataset_name='hpt_v41',
        dataset_type='torch',
        data_path='/export/share/yucheng/playground/hpt2.0/vila/hpt_v4_1_checked.json',
        # data_path='/export/share/yucheng/playground/hpt2.0/llava_v1_5_mix665k_test.json',
        image_path='/export/share/dataset/llava1.6'
    )
    add_dataset(hpt_v41)
    
    hpto_v0 = Dataset(
        dataset_name='hpto_v0',
        dataset_type='torch',
        data_path='/export/share/yucheng/playground/hpt2.0/hpto_fuse/hpto_v0_checked.json',
        image_path='/export/share/dataset/llava1.6'
    )
    add_dataset(hpto_v0)
    
    doc_downstream = Dataset(
        dataset_name='doc_downstream',
        dataset_type='torch',
        data_path='/export/share/yucheng/playground/hpt2.0/ocr_related/doc_downstream_conversations_checked.json', # normal
        image_path='/export/share/dataset/llava1.6'
    )
    add_dataset(doc_downstream)
    
    doc_reason25k = Dataset(
        dataset_name='doc_reason25k',
        dataset_type='torch',
        data_path='/export/share/yucheng/playground/hpt2.0/ocr_related/doc_reason25k_checked.json',
        image_path='/export/share/dataset/llava1.6'
    )
    add_dataset(doc_reason25k)
    
    doc_local4k = Dataset(
        dataset_name='doc_local4k',
        dataset_type='torch',
        data_path='/export/share/yucheng/playground/hpt2.0/ocr_related/doc_local4k_checked.json',
        image_path='/export/share/dataset/llava1.6'
    )
    add_dataset(doc_local4k)
    
    # Others
    sharegpt4v_gpt4_100k = Dataset(
        dataset_name="sharegpt4v_gpt4_100k",
        dataset_type="torch",
        data_path="./playground/datasharegpt_video/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json",
        image_path="./playground/datasharegpt_video/ShareGPT4V/data",
    )
    add_dataset(sharegpt4v_gpt4_100k)

    llava_instruct = Dataset(
        dataset_name="llava_instruct",
        dataset_type="torch",
        data_path="./playground/data/llava_instruct_150k_zh.jsonl",
        image_path="./playground/data/coco",
        description="",
    )
    add_dataset(llava_instruct)

    sharegpt4v_sft = Dataset(
        dataset_name='sharegpt4v_sft',
        dataset_type='torch',
        data_path='./playground/data/sharegpt4v/sharegpt4v_mix738k_remove_sa.json',
        image_path='./playground/data'
    )
    add_dataset(sharegpt4v_sft)

    dvqa_train_200k = Dataset(
        dataset_name="dvqa_train_200k",
        dataset_type="torch",
        data_path="./playground/data/dvqa_train_200k.jsonl",
        image_path="./playground/data/dvqa",
        description="",
    )
    add_dataset(dvqa_train_200k)

    chartqa_train_18k = Dataset(
        dataset_name="chartqa_train_18k",
        dataset_type="torch",
        data_path="./playground/data/chartqa_train_18k.jsonl",
        image_path="./playground/data/chartqa",
        description="",
    )
    add_dataset(chartqa_train_18k)

    ai2d_train_12k = Dataset(
        dataset_name="ai2d_train_12k",
        dataset_type="torch",
        data_path="./playground/data/ai2d_train_12k.jsonl",
        image_path="./playground/data/ai2d",
        description="",
    )
    add_dataset(ai2d_train_12k)

    docvqa_train_10k = Dataset(
        dataset_name="docvqa_train_10k",
        dataset_type="torch",
        data_path="./playground/data/docvqa_train_10k.jsonl",
        image_path="./playground/data/docvqa",
        description="",
    )
    add_dataset(docvqa_train_10k)

    geoqa = Dataset(
        dataset_name="geoqa",
        dataset_type="torch",
        data_path="./playground/data/geoqa+.jsonl",
        image_path="./playground/data/geoqa+",
        description="",
    )
    add_dataset(geoqa)

    synthdog_en = Dataset(
        dataset_name="synthdog_en",
        dataset_type="torch",
        data_path="./playground/data/synthdog_en.jsonl",
        image_path="./playground/data/synthdog-en",
        description="",
    )
    add_dataset(synthdog_en)

    vflan = Dataset(
        dataset_name='vflan',
        dataset_type='vflan',
        data_path='./playground/data/vlm-flan-clean-text1m-nosqa-sharded'
    )
    add_dataset(vflan)


    scienceqa = Dataset(
        dataset_name="scienceqa",
        dataset_type="torch",
        data_path="./playground/data/scienceqa/scienceqa_train_12k.json",
        image_path="./playground/data/scienceqa/images",
    )
    add_dataset(scienceqa)

    
    sherlock = Dataset(
        dataset_name="sherlock",
        dataset_type="torch",
        data_path="./playground/data/sherlock/processed/sherlock_317k.json",
        image_path="./playground/data/sherlock/images",
    )
    add_dataset(sherlock)
    math = Dataset(
        dataset_name="math",
        dataset_type="vflan",
        data_path="./playground/data/math",
    )
    add_dataset(math)

    wit_subset = Dataset(
        dataset_name="wit_subset",
        dataset_type="torch",
        data_path="./playground/data/WIT/wit_1_8m/wit_processed_538k.json",
        image_path="./playground/data/WIT/wit_1_8m/images"
    )
    add_dataset(wit_subset)

    youcook2 = Dataset(
        dataset_name="youcook2",
        dataset_type="torch",
        data_path="./playground/data/youcook2/youcook_filtered_v3.json",
        image_path="./playground/data/youcook2/video_data_clipped",
    )
    add_dataset(youcook2)
    
    vatex = Dataset(
        dataset_name="vatex",
        dataset_type="torch",
        data_path="/export/share/dataset/vatex/vatex/vatex_filtered_v3_checked.json",
        image_path="/export/share/dataset/vatex/train",
    )
    add_dataset(vatex)

    video_chatgpt = Dataset(
        dataset_name="video_chatgpt",
        dataset_type="torch",
        data_path="./playground/data/Video_ChatGPT/VideoInstruct-100K/VideoInstruct100K.json",
        image_path="./playground/data/Video_ChatGPT/activitynet_videos/",
    )
    add_dataset(video_chatgpt)

    shot2story_shotonly = Dataset(
        dataset_name="shot2story_shotonly",
        dataset_type="torch",
        data_path="./playground/data/shot2story/shot2story_shotonly.json",
        image_path="./playground/data/shot2story/Shot2Story/data/videos_extracted",
    )
    add_dataset(shot2story_shotonly)

    sharegpt_video = Dataset(
        dataset_name="sharegpt_video",
        dataset_type="torch",
        data_path="./playground/data/sharegpt_video/video_caption_pretrain.json",
        image_path="./playground/data/sharegpt_video/videos",
    )
    add_dataset(sharegpt_video)