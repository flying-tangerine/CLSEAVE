import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import ChineseCLIPConfig

class CLIPVEmodel(nn.Module):
    def __init__(self, clip, input_dim, hidden_size1, hidden_size2):
        super().__init__()
        self.clip = clip
        self.input_size = input_dim * 4
        output_size = 3
        
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("bn0", nn.BatchNorm1d(num_features=self.input_size)),
        #     ("drop0", nn.Dropout(p=0.1)),
        #     ("fc1", nn.Linear(in_features=self.input_size, out_features=hidden_size1)),
        #     ("rl1", nn.ReLU()),
        #     ("fc2", nn.Linear(in_features=hidden_size1, out_features=hidden_size2)),
        #     ("rl2", nn.ReLU()),
        #     ("bn1", nn.BatchNorm1d(num_features=hidden_size2)),
        #     ("drop1", nn.Dropout(p=0.1)),
        #     ("fc3", nn.Linear(in_features=hidden_size2, out_features=output_size))
        # ]))
        # config = ChineseCLIPConfig.from_pretrained(student_model_name)
        config = ChineseCLIPConfig()
        self.projection_dim = config.projection_dim # 512
        self.text_embed_dim = config.text_config.hidden_size # 768
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.bn0 = nn.BatchNorm1d(num_features=self.input_size)
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=hidden_size1)
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size2)
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=output_size)

    def forward(self, image_features, texts):
        text_outputs = self.clip(texts, return_dict=True) # output_hidden_states=True
        print(text_outputs[0])
        pooled_output = text_outputs[0][:, 0, :]
        text_features = self.text_projection(pooled_output)

        # pooled_output = text_outputs[1]
        # pooled_output = self.text_projection(pooled_output)

        image_features = image_features.view(image_features.size(0), -1)
        text_features = text_features.view(text_features.size(0), -1)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True) # normalization
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
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
