import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class student(nn.Module):
    def __init__(self, config: BertConfig, student_model_name):
        super(student, self).__init__()
        self.config = config
        embed_dim = config.hidden_size # 768
        projection_dim = 512
        self.mbert = BertModel.from_pretrained(student_model_name, add_pooling_layer=False)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # mbert: 1e-12, clip_vit: 1e-5
        self.text_projection = nn.Linear(embed_dim, projection_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.mbert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) # torch.Size([128, 28, 768])
        last_hidden_state = self.final_layer_norm(outputs[0]) # normalize the last hidden states            # torch.Size([128, 28, 768])
        pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]                                                                                          # torch.Size([128, 768])
        text_embeds = self.text_projection(pooled_output)                                                   # torch.Size([128, 512])
        return text_embeds, outputs.hidden_states

class mlp(nn.Module):
    def __init__(self,input_dim, hidden_size1, hidden_size2):
        super().__init__()
        self.input_size = input_dim * 4
        output_size = 3
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.bn0 = nn.BatchNorm1d(num_features=self.input_size)
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=hidden_size1)
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size2)
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=output_size)
    
    def forward(self, x):
        x = self.dropout(self.bn0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(self.bn1(x))
        output = self.fc3(x)
        return output

class CLIPVEmodel(nn.Module):
    def __init__(self, clicoteackpt, clipve_ckpt):
        super().__init__()
        
        self.embed = student(BertConfig(), 'bert-base-multilingual-cased')
        student_state_dict = {
            k.replace("student_model.", ""): v
            for k, v in torch.load(clicoteackpt, map_location=torch.device('cpu'))["model"].items()
            if k.startswith("student_model.")
        }
        self.embed.load_state_dict(student_state_dict) # , strict=False

        self.mlp = mlp(512, 128, 128)
        mlp_state_dict = {  
            k: v
            for k, v in torch.load(clipve_ckpt, map_location=torch.device('cpu'))['model_state_dict'].items()
            if not k.startswith("clip.")
        }
        self.mlp.load_state_dict(mlp_state_dict)

        for param in self.embed.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters(): # comment for the mlp fine-tuning
            param.requires_grad = False
        
        
    def forward(self, image_features, texts, atten_mask):
        with torch.no_grad():
            text_features, _ = self.embed(input_ids=texts, attention_mask=atten_mask)
        
        image_features = image_features.view(image_features.size(0), -1)
        text_features = text_features.view(text_features.size(0), -1)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) # normalization
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        image_features = self.mlp.dropout(image_features)
        text_features = self.mlp.dropout(text_features)
        
        combined_features = torch.cat([image_features, text_features, torch.abs(image_features - text_features), image_features * text_features], dim=1)

        output = self.mlp(combined_features)
        return output
