# encoding: utf-8
import torch
import torch.nn as nn
from .CLIPVEmodel import CLIPVEmodel
from transformers import BertConfig, BertModel, CLIPTextModelWithProjection


class student(nn.Module):
    def __init__(self, config: BertConfig, student_model_name):
        super(student, self).__init__()
        self.config = config
        self.eos_token_id = 102
        embed_dim = config.hidden_size # 768
        projection_dim = 512
        self.mbert = BertModel.from_pretrained(student_model_name, add_pooling_layer=False)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # mbert: 1e-12, clip_vit: 1e-5
        self.text_projection = nn.Linear(embed_dim, projection_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.mbert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) # torch.Size([128, 28, 768])
        last_hidden_state = self.final_layer_norm(outputs[0]) # normalize the last hidden states            # torch.Size([128, 28, 768])
        # pooled_output = last_hidden_state[
        #         torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        #         input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        #     ]                                                                                               # torch.Size([128, 768])
        pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]                                                                                        
        text_embeds = self.text_projection(pooled_output)                                                   # torch.Size([128, 512])
        return text_embeds, outputs.hidden_states


class TokenAlignmentModel(nn.Module):
    def __init__(
        self,
        teacher_model_name: str = 'openai/clip-vit-base-patch16',
        student_model_name: str = 'google-bert/bert-base-multilingual-cased',
        num_layers: int = 1,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        
        textmodel = CLIPTextModelWithProjection.from_pretrained(teacher_model_name)
        teacher_model = CLIPVEmodel(textmodel, 512, 128, 128)
        checkpoint = torch.load('/Users/ziyixu/Documents/masterThesis/CLIPCoTEA/mclip_text_3.pt')
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        self.teacher_model = teacher_model.to(device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        config = BertConfig()
        self.student_model = student(config, student_model_name).to(device)

        self.tch_embedding_size = 512
        self.stu_embedding_size = 512 # mbert: 768, clip_vit: 512

    def train(self, mode: bool = True):
        self.teacher_model.eval()
        self.student_model.train()

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
        device = src_input_ids.device

        with torch.no_grad(): # 在此处只使用text encoder来处理token_ids
            src_outputs = self.teacher_model.clip(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                output_hidden_states=True,
            )
        text_embeds, hidden_states = self.student_model(tgt_input_ids, tgt_attention_mask)

        return src_outputs.text_embeds, text_embeds

