from tqdm import tqdm
import clip
import torch

def train_model(model, train_dloader, train_dset,
                val_dloader, val_dset, 
                optimizer, loss, device, 
                epochs=10, early_stopper=None):
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(epochs):
        train_loss, train_accuracy = train_eva(model, train_dloader, train_dset, optimizer, loss, device)
        val_loss, val_accuracy = validate_eva(model, val_dloader, val_dset, loss, device)

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if early_stopper is not None and early_stopper.early_stop(val_loss):
            print("Early stopping triggered.")
            break
        # torch.cuda.empty_cache()

    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

def train_epoch(model, dataloader, dataset, optimizer, loss, device):
    model.train()
    total_loss = 0.0
    num_correct = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        images, texts, labels = batch
        images = images.to(torch.float32).to(device) # to(torch.float32)
        texts = texts.to(device)
        labels = labels.to(device)

        preds = model(images, texts)
        print(labels, labels.size())
        print(preds, preds.size())
        batch_loss = loss(preds, labels)
        print(batch_loss.item(), batch_loss.size())
        batch_loss.backward()
        optimizer.step()
        # if device == "cpu":
        #     optimizer.step()
        # else: 
        #     model.float()
        #     optimizer.step()
        #     clip.model.convert_weights(model)

        total_loss += batch_loss.item()
        num_correct += (preds.argmax(1) == labels).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = num_correct / len(dataset)

    return average_loss, accuracy

def train_eva(model, dataloader, dataset, optimizer, loss, device):
    model.train()
    total_loss = 0.0
    num_correct = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        images, texts, atten_mask, labels = batch
        images = images.to(torch.float32).to(device) # to(torch.float32)
        texts = texts.to(device)
        atten_mask = atten_mask.to(device)
        labels = labels.to(device)

        preds = model(images, texts, atten_mask)
        batch_loss = loss(preds, labels)
        # print(batch_loss.item())
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        num_correct += (preds.argmax(1) == labels).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = num_correct / len(dataset)

    return average_loss, accuracy

def validate(model, dataloader, dataset, loss, device):
    model.eval()
    total_loss = 0.0
    num_correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images, texts, labels = batch
            images = images.to(torch.float32).to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            preds = model(images, texts)
            batch_loss = loss(preds, labels)

            total_loss += batch_loss.item()
            num_correct += (preds.argmax(1) == labels).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = num_correct / len(dataset)

    return average_loss, accuracy

def validate_eva(model, dataloader, dataset, loss, device):
    model.eval()
    total_loss = 0.0
    num_correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images, texts, atten_mask, labels = batch
            images = images.to(torch.float32).to(device)
            texts = texts.to(device)
            atten_mask = atten_mask.to(device)
            labels = labels.to(device)

            preds = model(images, texts, atten_mask)
            batch_loss = loss(preds, labels)

            total_loss += batch_loss.item()
            num_correct += (preds.argmax(1) == labels).sum().item()

    average_loss = total_loss / len(dataloader)
    accuracy = num_correct / len(dataset)

    return average_loss, accuracy


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
