from textfeatureengineer import TextFeatureEngineer, WordTokenizer, SentenceTokenizer


from arguments import Arguments
import os
from transformers import (
    Trainer,
    default_data_collator,
    AutoModelForCausalLM,
    set_seed,
    AutoTokenizer,
    AutoConfig,
    
)
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from itertools import chain
from argparse import ArgumentParser
import pdb
import pandas as pd
from importlib import reload
from transformers import GPT2LMHeadModel
class GPT2LMHeadModel(GPT2LMHeadModel):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, text=None, **kwargs):
        if text is not None:
            input_ids = torch.cat([input_ids, text], dim=1)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, **kwargs)

class GenerationModelTrainer():
    def __init__(self, local_path, file_path, key_name = "generation_model_1"):
        super().__init__()
        self.arguments = Arguments(output_dir = local_path)
        self.file_path = file_path
        #self.arguments = Arguments(output_dir = self.local_path)
        #local_path

    def load_lm_dataset(self, df_path, word_tokenizer, data_training_args):
        """ df_path data load
        """
        
        def group_texts(examples, block_size = data_training_args.block_size):
            """ block_size만큼 padding
            """
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        data_files = {}
        #pdb.set_trace()
        data_files["train"] = df_path
        datasets = load_dataset('csv', data_files = data_files, cache_dir='/content/drive/MyDrive/CJ ONs/cache')
        datasets["validation"] = load_dataset('csv',
        data_files=data_files,
        split=f"train[:{data_training_args.validation_split_percentage}%]",
        cache_dir='/content/drive/MyDrive/CJ ONs/cache')
        
        datasets["train"] = load_dataset(
            'csv',
            data_files=data_files,
            split=f"train[{data_training_args.validation_split_percentage}%:]",
            cache_dir='/content/drive/MyDrive/CJ ONs/cache',
        )
        column_names = datasets["train"].column_names
        print(data_training_args)
        tokenized_datasets = datasets.map(
            word_tokenizer.tokenizing_df,
        )
        lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=True)
        return lm_datasets

    
    def make_model(self, model_args):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        return model
    # def make_tokenizer(self, model_args):
        
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

        
    #     return tokenizer 
    def train(self, lm_datasets, tokenizer, model, training_args):
        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]
        pdb.set_trace()
        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        )
        
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )

        set_seed(training_args.seed)

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        pdb.set_trace()
        trainer.train(resume_from_checkpoint=checkpoint)
        return trainer

    def run(self, runtype = 'train'):
        # prepare data
        text_feature_engineer = TextFeatureEngineer()
        df_path = self.file_path
        df = text_feature_engineer.load_dataset(df_path)
        # #df_path_filtered = 'datas/data_tmp.csv'
        # #pdb.set_trace()
        df = text_feature_engineer.preprocess(df)
        filtered_path = '/content/drive/MyDrive/CJ ONs/prac_ppver.csv'
        df.to_csv(filtered_path,index=False)

        if runtype == 'train':
            word_tokenizer = WordTokenizer()
            
            # load dataset and model
            lm_datasets = self.load_lm_dataset('/content/drive/MyDrive/CJ ONs/prac_ppver.csv', word_tokenizer, data_training_args = self.arguments.data_training_args)
            print("lm_datasets",lm_datasets)
            print("self.arguments.model_args",self.arguments.model_args)
            model = self.make_model(model_args = self.arguments.model_args)
            tokenizer = word_tokenizer.tokenizer
            
            # train
            # pdb.set_trace()
            trainer = self.train(lm_datasets, tokenizer, model, training_args = self.arguments.training_args)

            # save model
            trainer.save_model()
    
    
        elif runtype == 'embedding':
            
            # load embedder
            sentence_tokenizer = SentenceTokenizer(device='cuda')

            # embedding
            sentence_tokenizer.run(df, method='sentence')

if __name__ == '__main__':
    parser = ArgumentParser("train or embedding")
    
    parser.add_argument("--runtype", help="train or embedding", default = 'train', type=str)
    parser.add_argument("--model_key_name", help="model_key_name", default = 'generation', type=str)
    parser.add_argument("--file_path", help="file_path", default = '/content/drive/MyDrive/CJ ONs/prac.csv', type=str)
    parser.add_argument("--model_save_path", help="model_save_path", default = '/content/drive/MyDrive/CJ ONs/기존 모델/model_original', type=str)
    args = parser.parse_args()
    
    generation_trainer = GenerationModelTrainer(key_name = args.model_key_name, local_path = args.model_save_path, file_path=args.file_path)

    generation_trainer.run(runtype = args.runtype)