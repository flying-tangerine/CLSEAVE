from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NLIVEdataset(Dataset):
    def __init__(self, image_folder, df, device, model, preprocess):
        super().__init__()
        self.image_folder = image_folder
        self.image_id = df["Flickr30K_ID"].tolist()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.text = df["sentence2"].tolist()
        self.label = df["gold_label"].tolist()
        self.device = device
        self.model = model
        self.preprocess = preprocess

    def clip_collate(self, batch_list, model):
        assert type(batch_list) == list, f"Error"
        images = torch.stack([item[0] for item in batch_list], dim=0).to(self.device)
        texts = torch.stack([item[1] for item in batch_list], dim=0).to(self.device)
        masks = torch.stack([item[2] for item in batch_list], dim=0).to(self.device)
        labels = torch.LongTensor([item[3] for item in batch_list]).to(self.device)

        image_features = model.encode_image(images)
    #     text_features = model.encode_text(texts)
        return image_features, texts, masks, labels

    def get_loader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=lambda x: self.clip_collate(x, self.model)
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        image_id = self.image_id[idx]
        image = self.preprocess(Image.open(f"{self.image_folder}/{image_id}.jpg")) # Image from PIL module
        tokens = self.tokenizer(self.text[idx], padding='max_length', truncation=True, max_length=77, return_tensors="pt")
        token_ids_tensor = tokens['input_ids'][0]
        text = token_ids_tensor.to(torch.int32)
        atten_mask = tokens['attention_mask'][0]
        atten_mask = atten_mask.to(torch.int32)
        label = self.label[idx]
        return image,text,atten_mask,label