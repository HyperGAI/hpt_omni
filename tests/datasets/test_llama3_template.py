import json
import copy
from transformers import AutoTokenizer, AutoConfig

from llava import conversation as conversation_lib
from llava.train.args import DataArguments, TrainingArguments
from llava.data.dataset import preprocess_llama_3, LazySupervisedDataset, preprocess_multimodal,DataCollatorForSupervisedDataset, LazyMMC4Dataset
from llava.model.multimodal_encoder.intern_encoder import InternVisionPreprocessor


class TestupervisedDataset():
    def test_preprocess_llama_3(self):
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3_fix"]
        llm_path = '/export/share/models/Meta-Llama-3-8B-Instruct/'
        
        llm_cfg = AutoConfig.from_pretrained(llm_path)
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, 
            model_max_length=llm_cfg.max_position_embeddings,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        data_args = DataArguments(
            is_multimodal=True,
        )
        # TODO: should set False in this function, can put after tokenization
        data_args.mm_use_im_start_end = False
        
        data_path = "/export/share/yucheng/playground/hpt2.0/llava_v1_5_mix665k_test.json"
        image_folder = "/export/share/dataset/llava1.6"
        with open(data_path, "r") as fp:
            list_data_dict = json.load(fp)
        
        for i in range(10):
            sources = [list_data_dict[i]]
            # question = sources[0]['conversations'][0]['value']
            # question = question.replace('<image>', '').strip()
            # question = question + '\n<image>'
            # sources[0]['conversations'][0]['value'] = question
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), data_args)
            out_dict = preprocess_llama_3(sources=sources, tokenizer=tokenizer, has_image=True)
            breakpoint()
            # print (out_dict)
            '''
            conversations[0]:
            for llama3: 
            <|begin_of_text|>: 128000
            <|start_header_id|>: 128006
            <|end_header_id|>: 128007
            <|eot_id|>: 128009
            user: 882 (case sensitive)
            assistant: 78191
            system: 9125
            
            conversation:
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>

            <image>
            What are the colors of the bus in the image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            The bus in the image is white and red.<|eot_id|><|start_header_id|>user<|end_header_id|>

            What feature can be seen on the back of the bus?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            The back of the bus features an advertisement.<|eot_id|><|start_header_id|>user<|end_header_id|>

            Is the bus driving down the street or pulled off to the side?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            The bus is driving down the street, which is crowded with people and other vehicles.<|eot_id|>
            
            input_ids:
            tensor([[128000, 128006,   9125, 128007,    271,   2675,    527,    264,  11190,
                    4221,    323,  11376,  18328,     13,   1472,    527,   3025,    311,
                    3619,    279,   9302,   2262,    430,    279,   1217,   5825,     11,
                        323,   7945,    279,   1217,    449,    264,   8205,    315,   9256,
                    1701,   5933,   4221,     13, 128009, 128006,    882, 128007,    271,
                    -200,   3923,    527,    279,   8146,    315,    279,   5951,    304,
                        279,   2217,     30, 128009, 128006,  78191, 128007,    271,    791,
                    5951,    304,    279,   2217,    374,   4251,    323,   2579,     13,
                    128009, 128006,    882, 128007,    271,   3923,   4668,    649,    387,
                    3970,    389,    279,   1203,    315,    279,   5951,     30, 128009,
                    128006,  78191, 128007,    271,    791,   1203,    315,    279,   5951,
                    4519,    459,  33789,     13, 128009, 128006,    882, 128007,    271,
                    3957,    279,   5951,  10043,   1523,    279,   8761,    477,  13541,
                    1022,    311,    279,   3185,     30, 128009, 128006,  78191, 128007,
                        271,    791,   5951,    374,  10043,   1523,    279,   8761,     11,
                        902,    374,  39313,    449,   1274,    323,   1023,  11731,     13,
                    128009]])
                    
            targets:
            tensor([[  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,    791,
                    5951,    304,    279,   2217,    374,   4251,    323,   2579,     13,
                    128009,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,    791,   1203,    315,    279,   5951,
                    4519,    459,  33789,     13, 128009,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                    -100,    791,   5951,    374,  10043,   1523,    279,   8761,     11,
                        902,    374,  39313,    449,   1274,    323,   1023,  11731,     13,
                    128009]])
                    
            target_useful = input_ids[0,torch.where(targets>0)[1]]
            tokenizer.decode(target_useful):
            'The bus in the image is white and red.<|eot_id|>The back of the bus features an advertisement.<|eot_id|>The bus is driving down the street, which is crowded with people and other vehicles.<|eot_id|>'
            '''
        
        data_args.image_processor = InternVisionPreprocessor()
        dataset = LazySupervisedDataset(
            data_path=data_path,
            image_folder=image_folder,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=None
        )
        for i in range(10):
            x = dataset.__getitem__(i)
            print(x)
            
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        
    
    def test_preprocess_llama_3_interleaved(self):
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3_fix"]
        llm_path = '/export/share/models/Meta-Llama-3-8B-Instruct/'
        
        llm_cfg = AutoConfig.from_pretrained(llm_path)
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, 
            model_max_length=llm_cfg.max_position_embeddings,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        train_args = TrainingArguments(output_dir='')
        
        data_args = DataArguments(
            is_multimodal=True,
        )
        data_args.image_processor = InternVisionPreprocessor()
        dataset = LazyMMC4Dataset(
            data_path='/export/share/dataset/MMC4/pkl_test',
            image_folder=None,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=train_args
        )
        
        for i in range(10):
            x = dataset.__getitem__(i)
            print (x)
        
    

if __name__ == '__main__':
    test = TestupervisedDataset()
    # test.test_preprocess_llama_3()
    test.test_preprocess_llama_3_interleaved()