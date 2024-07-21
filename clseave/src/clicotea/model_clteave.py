# encoding: utf-8
import torch
import torch.nn as nn
from .CLIPVEmodel import CLIPVEmodel, AE
from transformers import BertConfig, BertModel, CLIPTextModelWithProjection

def apply_same_dropout(src_hidden_states, tgt_hidden_states, dropout_rate):
    dropout_mask = (torch.rand_like(src_hidden_states) > dropout_rate).float()
    dropout_mask = dropout_mask / (1.0 - dropout_rate)  # Scale mask to keep expected values the same

    src_hidden_states = src_hidden_states * dropout_mask
    tgt_hidden_states = tgt_hidden_states * dropout_mask

    return src_hidden_states, tgt_hidden_states

class TokenAlignmentModel(nn.Module):
    def __init__(
        self,
        teacher_model_name: str = "/root/autodl-tmp/CliCoTea/vitb16",
        student_model_name: str = "/root/autodl-tmp/CliCoTea/mbert",
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        textmodel = CLIPTextModelWithProjection.from_pretrained(teacher_model_name)
        teacher_model = CLIPVEmodel(textmodel, 512, 128, 128)
        checkpoint = torch.load('/root/autodl-tmp/CliCoTea/src/clicotea/mclip_text_3.pt')
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        self.teacher_model = teacher_model.to(device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        student_config = BertConfig.from_pretrained(student_model_name)
        student_config.num_hidden_layers = 12 # albef: 6, clip: 12
        self.student_model = self.create_student_model(student_model_name).to(device)
        
        AE_model = AE()
        checkpoint = torch.load('/root/autodl-tmp/CliCoTea/src/clicotea/autoencoder_1.pt') # 1: multilingual
        AE_model.load_state_dict(checkpoint['model_state_dict'])
        self.AE_model = AE_model.to(device)

        self.tch_embedding_size = 512
        self.stu_embedding_size = student_config.hidden_size # mbert: 768, clip_vit: 512

    def train(self, mode: bool = True):
        self.teacher_model.eval()
        self.student_model.train()
        self.AE_model.train()

    def forward(self, inputs): #这个input是从哪里输入哪里调用的？-> dataset
        (
            src_input_ids,
            tgt_input_ids,
            src_attention_mask,
            tgt_attention_mask,
            src_idx,
            tgt_idx,
            _,
        ) = inputs
        assert len(src_idx) == len(tgt_idx)
        num_token = len(src_idx)
        device = src_input_ids.device

        with torch.no_grad(): # 在此处只使用text encoder来处理token_ids
            src_outputs = self.teacher_model.clip(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                output_hidden_states=True,
            )
        tgt_outputs = self.student_model(
            tgt_input_ids, attention_mask=tgt_attention_mask, output_hidden_states=True
        )

        src_token_hidden_states = torch.zeros(
            self.num_layers, num_token, self.tch_embedding_size, device=device
        )
        tgt_token_hidden_states = torch.zeros(
            self.num_layers, num_token, self.stu_embedding_size, device=device
        )
        for i, layer in enumerate(range(-1, -1 - self.num_layers, -1)):
            src_token_hidden_states[i] = torch.index_select(
                src_outputs.hidden_states[layer].view(-1, self.tch_embedding_size),
                0,
                src_idx,
            )
            tgt_token_hidden_states[i] = torch.index_select(
                tgt_outputs.hidden_states[layer][:self.stu_embedding_size].view(-1, self.stu_embedding_size),
                0,
                tgt_idx,
            )
        
        tgt_token_hidden_states = self.AE_model.encoder(tgt_token_hidden_states)
#         src_token_hidden_states, tgt_token_hidden_states = apply_same_dropout(
#             src_token_hidden_states, tgt_token_hidden_states, self.dropout_rate
#         )
        
        return src_token_hidden_states, tgt_token_hidden_states 

    @staticmethod
    def create_student_model(student_model_name):
        student_config = BertConfig.from_pretrained(student_model_name)
        student_config.num_hidden_layers = 12
        return BertModel.from_pretrained(
            student_model_name, config=student_config, add_pooling_layer=False
        )
