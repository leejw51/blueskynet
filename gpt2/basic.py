from transformers import AutoModel
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

model_name = "bert-base-cased"
model = AutoModel.from_pretrained(model_name)
