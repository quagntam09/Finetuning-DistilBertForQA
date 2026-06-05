import torch.nn as nn
from transformers import DistilBertModel
from peft import LoraConfig, get_peft_model
from config_model import Config
class CustomLoraDistilBertQA(nn.Module):
    def __init__(self):
        super(CustomLoraDistilBertQA, self).__init__()
        self.config = Config.from_yaml()
        self.basemodel = DistilBertModel.from_pretrained(self.config.model_name)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias=self.config.lora_bias
        )
        self.distilbert_lora = get_peft_model(self.basemodel, lora_config)

        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.relu = nn.ReLU()

        self.qa_outputs = nn.Linear(self.basemodel.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert_lora(input_ids = input_ids, attention_mask = attention_mask)
        hidden_state = outputs[0]

        x = self.dropout(hidden_state)
        x = self.relu(x)
        logits = self.qa_outputs(x)

        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

# Hàm nhỏ để in ra số lượng tham số thực sự cần train
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Tham số Trainable: {trainable_params:,} || Tổng tham số: {all_param:,} || Tỉ lệ: {100 * trainable_params / all_param:.2f}%")

if __name__ == "__main__":
    model = CustomLoraDistilBertQA()
    print_trainable_parameters(model)
