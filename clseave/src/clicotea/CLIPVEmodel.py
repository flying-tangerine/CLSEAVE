import torch
import torch.nn as nn

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

class CLIPVEmodel(nn.Module):
    def __init__(self, clip, input_dim, hidden_size1, hidden_size2):
        super().__init__()
        self.clip = clip
        self.input_size = input_dim * 4
        output_size = 3
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.bn0 = nn.BatchNorm1d(num_features=self.input_size)
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=hidden_size1)
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size2)
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=output_size)

    def forward(self, image_features, texts):
        output = self.clip(texts, return_dict=True) # using transformer clip
        text_features = output.text_embeds
        # text_features = self.clip.encode_text(texts) # directly load clip from github
        
        image_features = image_features.view(image_features.size(0), -1)
        text_features = text_features.view(text_features.size(0), -1)
        image_features = image_features / image_features.norm(dim=1, keepdim=True) # normalization
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = self.dropout(image_features)
        text_features = self.dropout(text_features)
        
        combined_features = torch.cat([image_features, text_features, torch.abs(image_features - text_features), image_features * text_features], dim=1)

        x = self.dropout(self.bn0(combined_features))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(self.bn1(x))
        output = self.fc3(x)
        # output = F.softmax(x, dim=1)
        return output # output.hidden_state