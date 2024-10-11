import torch
from transformers import AutoTokenizer, AutoModel


class Embedder:

    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.determine_best_device())

    @staticmethod
    def determine_best_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def embed(self, *text):
        inputs = self.tokenizer(list(text), return_tensors="pt", padding=True, truncation=True).to(
            self.determine_best_device())

        with torch.no_grad():
            outputs = self.model(**inputs)

        return torch.nn.functional.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1).cpu()
    
    def batch_embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
            self.determine_best_device())

        with torch.no_grad():
            outputs = self.model(**inputs)

        sentence_embeddings = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1).cpu()
        print(sentence_embeddings)
        return sentence_embeddings