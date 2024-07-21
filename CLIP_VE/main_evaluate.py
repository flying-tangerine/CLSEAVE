import clip
import torch
import pandas as pd
from CLIP_VE.CLIPCoTeamodel import CLIPVEmodel
from CLIP_VE.dataset_eva import NLIVEdataset
from CLIP_VE.utils import train_model, EarlyStopper, validate_eva
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-6
    image_folder = '/Users/ziyixu/SNLI-VE/data/flickr30k_images'
    clicoteackpt = '/Users/ziyixu/Documents/masterThesis/CLIPCoTEA/checkpoint_9.pt'
    # AEckpt = '/Users/ziyixu/Documents/masterThesis/CLIPCoTEA/autoencode_1.pt'
    clipve_ckpt = '/Users/ziyixu/Documents/masterThesis/CLIPCoTEA/mclip_text_3.pt'
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #[RN50, ViT-B/16, ViT-L/14], Must set jit=False for training

    df = pd.read_json('/Users/ziyixu/Documents/masterThesis/CliCoTea/data/snli_ve_train_de.json', lines=True)
    df = df[['Flickr30K_ID', 'sentence2', 'gold_label']]
    df.loc[:, "gold_label"] = df["gold_label"].replace({"entailment": 2, "neutral": 1, "contradiction": 0})
    df = df.sample(frac=1).reset_index(drop=True) # !make it full length [:18000]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) #, shuffle=True
    train_dataset = NLIVEdataset(image_folder, train_df, device, model, preprocess)
    val_dataset = NLIVEdataset(image_folder, val_df, device, model, preprocess)
    train_dataloader = train_dataset.get_loader(batch_size=BATCH_SIZE)
    val_dataloader = val_dataset.get_loader(batch_size = BATCH_SIZE)

    custom_model = CLIPVEmodel(clicoteackpt, clipve_ckpt).to(device)
    early_stopper = EarlyStopper(patience=1, min_delta=0)
    loss = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(custom_model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=LEARNING_RATE, weight_decay=0)

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = \
    train_model(custom_model, train_dataloader, train_dataset, 
            val_dataloader, val_dataset, optimizer, loss, device,
            epochs=EPOCHS, early_stopper=early_stopper)


    # -----test-----
    df_test = pd.read_json('/Users/ziyixu/Documents/masterThesis/CliCoTea/data/snli_ve_test_de.json', lines=True)
    df_test = df_test[['Flickr30K_ID', 'sentence2', 'gold_label']]
    df_test.loc[:, "gold_label"] = df_test["gold_label"].replace({"entailment": 2, "neutral": 1, "contradiction": 0})
    test_dataset = NLIVEdataset(image_folder, df_test, device, model, preprocess)
    test_dataloader = test_dataset.get_loader(batch_size=BATCH_SIZE)
    # for images, texts, labels in train_dataloader:
    #     print(f"image {images}{images.size(1)}, text {texts}{texts.size()}, labels: {labels}{labels.size()}")
    
#     custom_model = CLIPVEmodel(clicoteackpt, clipve_ckpt).to(device)
#     loss = torch.nn.CrossEntropyLoss().to(device)
    test_loss, test_accuracy = validate_eva(custom_model, test_dataloader, test_dataset, loss, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
