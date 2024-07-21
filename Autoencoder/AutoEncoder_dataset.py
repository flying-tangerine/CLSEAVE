# encoding: utf-8
import jsonlines
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertConfig, BertModel


class AEDataset(Dataset):
    def __init__(
        self,
        token_alignment_file: str,
        src_name: str = "openai/clip-vit-base-patch16",
        tgt_name: str = "bert-base-multilingual-cased",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.token_alignment_file = token_alignment_file
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_name)
        
        model_config = BertConfig.from_pretrained(tgt_name)
        self.embedding_size = model_config.hidden_size
        self.model = self.create_student_model(tgt_name).to(self.device)

        # define padding ids
        self.src_padding_id = self.src_tokenizer.convert_tokens_to_ids(
            self.src_tokenizer.pad_token
        )
        self.tgt_padding_id = self.src_tokenizer.convert_tokens_to_ids(
            self.src_tokenizer.pad_token
        )
        # load input file
        self.samples = []
        for obj in jsonlines.open(self.token_alignment_file):
            src_sent = obj["src"].strip().split()
            tgt_sent = obj["tgt"].strip().split()
            self.samples.append((src_sent, tgt_sent, obj["alignment"]))

    def collate_fn(self, batch):
        max_tgt = max([len(t["input_ids"]) for t, _ in batch])
        tgt_tensor = torch.zeros(len(batch), max_tgt, dtype=torch.long).fill_(
            self.tgt_padding_id
        ).to(self.device)
        tgt_attention_mask = torch.zeros(len(batch), max_tgt, dtype=torch.long).to(self.device)
        tgt_idx = []
        for i, (t, p) in enumerate(batch):
            tgt_tensor[i, : len(t["input_ids"])] = torch.tensor(
                t["input_ids"], dtype=torch.long
            ).to(self.device)
            tgt_attention_mask[i, : len(t["attention_mask"])] = torch.tensor(
                t["attention_mask"], dtype=torch.long
            ).to(self.device)
            tgt_idx += [b + i * max_tgt for _, b in p]
        tgt_idx = torch.tensor(tgt_idx, dtype=torch.long).to(self.device)

        num_token = len(tgt_idx)
        with torch.no_grad():
            tgt_outputs = self.model(tgt_tensor, attention_mask=tgt_attention_mask, output_hidden_states=True)
        tgt_token_hidden_states = torch.zeros(1, num_token, self.embedding_size).to(self.device)
        tgt_token_hidden_states = torch.index_select(
            tgt_outputs.hidden_states[-1].view(-1, self.embedding_size),
            0,
            tgt_idx,
        )
        return tgt_token_hidden_states

    def get_loader(self, batch_size, num_workers=0, train=False):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=train,
            collate_fn=self.collate_fn,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, align = self.samples[idx]
        src_tokens = self.src_tokenizer(src, is_split_into_words=True, truncation=True)
        tgt_tokens = self.tgt_tokenizer(tgt, is_split_into_words=True, truncation=True)
        src_word_ids = src_tokens.word_ids()
        tgt_word_ids = tgt_tokens.word_ids()
        word_pairs = [
            (src_word_ids.index(s), tgt_word_ids.index(t))  # take only left alignment
            for s, t in align
            if s in src_word_ids and t in tgt_word_ids
        ]
        return (tgt_tokens, word_pairs)

    @staticmethod
    def create_student_model(student_model_name):
        student_config = BertConfig.from_pretrained(student_model_name)
        student_config.num_hidden_layers = 12 # same as the mbert_base model
        return BertModel.from_pretrained(
            student_model_name, config=student_config, add_pooling_layer=False
        )
