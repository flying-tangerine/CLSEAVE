from tqdm import tqdm
import torch
import torch.nn as nn
from AutoEncoder_dataset import AEDataset


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 768),
            torch.nn.ReLU(),
#             torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        # input_size = [1, num_tokens, 768]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AE().to(device)
    train_dset = AEDataset(
        token_alignment_file="/Users/ziyixu/Documents/masterThesis/CliCoTea/data/snli/word_pairs_train_en-de.json",
        src_name="openai/clip-vit-base-patch16",
        tgt_name="bert-base-multilingual-cased",
        device = device,
    )
    train_loader = train_dset.get_loader(
        batch_size=BATCH_SIZE,
        num_workers=0, # 0
        train=True,
    )
    loss_function = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        total_loss = []
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            token_hidden_states = batch
            reconstructed = model(token_hidden_states)
            loss = loss_function(reconstructed, token_hidden_states)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        average_loss = sum(total_loss) / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {average_loss:.4f}")
    
    # torch.save({
    #     'epoch': EPOCHS,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': total_loss,
    #     }, f"/root/autodl-tmp/CliCoTea/src/clicotea/autoencoder.pt") 
