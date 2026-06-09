import torch.nn as nn
from transformers import DistilBertModel

from config_model import Config


class CustomDistilBertQA(nn.Module):
    """DistilBERT extractive QA model for full fine-tuning."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config.from_yaml()
        self.distilbert = DistilBertModel.from_pretrained(self.config.model_name)

        if getattr(self.config, "freeze_encoder", False):
            for param in self.distilbert.parameters():
                param.requires_grad = False

        hidden_size = self.distilbert.config.hidden_size
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        logits = self.qa_outputs(self.dropout(hidden_state))
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


# Backward-compatible alias for older scripts/imports.
CustomLoraDistilBertQA = CustomDistilBertQA


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    ratio = 100 * trainable_params / all_param if all_param else 0.0
    print(
        f"Tham số Trainable: {trainable_params:,} || "
        f"Tổng tham số: {all_param:,} || "
        f"Tỉ lệ: {ratio:.2f}%"
    )


if __name__ == "__main__":
    model = CustomDistilBertQA()
    print_trainable_parameters(model)
