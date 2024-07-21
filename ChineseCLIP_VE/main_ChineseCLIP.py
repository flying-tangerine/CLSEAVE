import cn_clip.clip as clip # import clip
import torch
import pandas as pd
from CLIP_VE.dataset import NLIVEdataset
from ChineseCLIPVEmodel import CLIPVEmodel
from CLIP_VE.utils import train_model, EarlyStopper, validate
from sklearn.model_selection import train_test_split
from transformers import ChineseCLIPTextModel # CLIPTextModelWithProjection, CLIPModel

df = pd.read_json('/Users/ziyixu/SNLI-VE/data/snli_ve_train.jsonl', lines=True) ##
# df = df[['Flickr30K_ID', 'sentence2', 'gold_label']]
df.loc[:, "gold_label"] = df["gold_label"].replace({"entailment": 2, "neutral": 1, "contradiction": 0})
df = df.sample(frac=1).reset_index(drop=True) # !make it full length [:18000]

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-6

model_name = "OFA-Sys/chinese-clip-vit-base-patch16" # "openai/clip-vit-base-patch16"
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load_from_name("ViT-B-16",device=device,download_root='./') #[RN50, ViT-B/16, ViT-L/14], Must set jit=False for training
textmodel = ChineseCLIPTextModel.from_pretrained(model_name, add_pooling_layer=False)

# for k in model.visual.transformer.parameters():
#     k.requires_grad=False
    
# if device == "cpu":
#     model.float()
# else :
#     clip.model.convert_weights(model)

image_folder = '/Users/ziyixu/SNLI-VE/data/flickr30k_images'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) #, shuffle=True
# train_df, val_df = torch.utils.data.random_split(df,[0.8, 0.2])
train_dataset = NLIVEdataset(image_folder, train_df, device, model, preprocess)
val_dataset = NLIVEdataset(image_folder, val_df, device, model, preprocess)

train_dataloader = train_dataset.get_loader(batch_size=BATCH_SIZE)
val_dataloader = val_dataset.get_loader(batch_size = BATCH_SIZE)

# for images, texts, labels in train_dataloader:
#     print(f"image {images}{images.size(1)}, text {texts}{texts.size()}, labels: {labels}{labels.size()}")
custom_model = CLIPVEmodel(textmodel, 512, 128, 128).to(device)
early_stopper = EarlyStopper(patience=1, min_delta=0)

loss = torch.nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(custom_model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer = torch.optim.AdamW(custom_model.parameters(), lr=LEARNING_RATE, weight_decay=0)

train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = \
    train_model(custom_model, train_dataloader, train_dataset, 
            val_dataloader, val_dataset, optimizer, loss, device,
            epochs=EPOCHS, early_stopper=early_stopper)


df_test = pd.read_json('/root/data/flickr30k_images/snli_ve_test.jsonl', lines=True)
df_test = df_test[['Flickr30K_ID', 'sentence2', 'gold_label']]
df_test.loc[:, "gold_label"] = df_test["gold_label"].replace({"entailment": 2, "neutral": 1, "contradiction": 0})

test_dataset = NLIVEdataset(image_folder, df_test, device, model, preprocess)
test_dataloader = test_dataset.get_loader(batch_size=BATCH_SIZE)
test_loss, test_accuracy = validate(custom_model, test_dataloader, test_dataset, loss, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

torch.save({
        'epoch': EPOCHS,
        'model_state_dict': custom_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss_history,
        'clip.text_model.embeddings.position_ids': torch.zeros(1, 77),
        }, f"data/mclip_text_2.pt") #just change to your preferred folder/filename