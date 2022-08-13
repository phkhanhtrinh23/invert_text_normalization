from datasets import load_dataset
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
import os
from transformers import EncoderDecoderModel
from datasets import DatasetDict

def get_sample_from_dataset(dataset, train_size=10000):
    dataset_small = DatasetDict()
    dataset_small['valid'] = dataset['valid']
    dataset_small['test'] = dataset['test']
    dataset_small['train'] = dataset['train'].shuffle(seed=0).select(range(train_size))
    return dataset_small

def download_tokenizer_files():
    resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
    for item in resources:
        if not os.path.exists(os.path.join(cache_dir, item)):
            tmp_file = hf_bucket_url(model_name, filename=item)
            tmp_file = cached_path(tmp_file, cache_dir=cache_dir)
            os.rename(tmp_file, os.path.join(cache_dir, item))

def init_tokenizer():
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    return tokenizer

def init_model(cache_dir, model_name):
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)

    # set encoder decoder tying to True
    roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name,
                                                                         model_name,
                                                                         tie_encoder_decoder=False)

    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    roberta_shared.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    # set decoding params
    roberta_shared.config.max_length = 512
    roberta_shared.config.early_stopping = True
    roberta_shared.config.no_repeat_ngram_size = 3
    roberta_shared.config.length_penalty = 2.0
    roberta_shared.config.num_beams = 1
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    return roberta_shared, tokenizer

if __name__ == "__main__":
    dataset = load_dataset('VietAI/spoken_norm_assignment')
    cache_dir = './cache'
    model_name = 'nguyenvulebinh/envibert'

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    model, tokenizer = init_model(cache_dir, model_name)

    dataset = get_sample_from_dataset(dataset, train_size=400000)
    new_dataset = dataset.map(lambda example: {"src": " ".join(example["src"])}, remove_columns=["src"])
    dataset = new_dataset.map(lambda example: {"tgt": " ".join(example["tgt"])}, remove_columns=["tgt"])

    device = "cuda:1"
    trained_model = model.from_pretrained('./checkpoints/checkpoint-225000').to(device)

    with open("output.txt", "w", encoding="utf-8") as f:
        for data in dataset["test"]:
            input_ids = tokenizer.encode(data["src"], return_tensors='pt', truncation=True, max_length=512, padding="max_length").to(device)
            beam_outputs = trained_model.generate(
                input_ids, 
                max_length=512, 
                num_beams=5,
                no_repeat_ngram_size=2, 
                num_return_sequences=1, 
                early_stopping=True
            )
            output_pieces = tokenizer.convert_ids_to_tokens(beam_outputs[0].cpu().numpy().tolist())
            output_text = tokenizer.sp_model.decode(output_pieces).replace("<pad>","")
            f.write(output_text)
            f.write("\n")
    f.close()