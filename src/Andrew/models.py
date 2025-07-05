import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTokenizer, AutoTokenizer, AutoModelForCausalLM

class CLIPCaptioningModel(nn.Module):
    def __init__(self,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 decoder_hidden_dim: int = 512,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 vocab_size: int = None,
                 max_length: int = 32):
        super().__init__()
        # 1) Load CLIP’s vision encoder
        self.clip_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        # freeze CLIP weights if you only want to train the decoder:
        for param in self.clip_encoder.parameters(): param.requires_grad = False

        # 2) Tokenizer / embeddings for your decoder
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        if vocab_size is None:
            vocab_size = self.tokenizer.vocab_size
        self.token_embedding = nn.Embedding(vocab_size, decoder_hidden_dim)

        # 3) Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, decoder_hidden_dim))

        # 4) TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_hidden_dim,
            nhead=num_heads,
            dim_feedforward=decoder_hidden_dim * 4,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # 5) Final output projection to vocab
        self.output_proj = nn.Linear(decoder_hidden_dim, vocab_size)

        self.max_length = max_length

    def forward(self, pixel_values: torch.FloatTensor, 
                      input_ids: torch.LongTensor):
        """
        pixel_values: torch.FloatTensor of shape (B, 3, H, W) – preprocessed by CLIPProcessor
        input_ids:     torch.LongTensor of shape (B, T) – token IDs for teacher forcing
        """
        # Encode images
        vision_outputs = self.clip_encoder(pixel_values)
        # take the last hidden states of the [CLS] token from the vision transformer
        # shape (B, hidden_dim)
        img_feats = vision_outputs.pooler_output  
        
        # Expand to sequence length: (B, 1, hidden_dim)
        memory = img_feats.unsqueeze(1)

        # Prepare decoder inputs
        B, T = input_ids.shape
        token_embeds = self.token_embedding(input_ids)               # (B, T, D)
        token_embeds = token_embeds + self.pos_embedding[:, :T, :]  # add pos encoding
        # Transformer expects (T, B, D)
        tgt = token_embeds.permute(1, 0, 2)
        # Memory must be (S, B, D), here S=1
        memory = memory.permute(1, 0, 2)

        # Causal mask so each position can only attend to earlier ones
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(input_ids.device)

        # Decode
        decoded = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=causal_mask
        )  # (T, B, D)

        decoded = decoded.permute(1, 0, 2)  # (B, T, D)
        logits = self.output_proj(decoded)  # (B, T, vocab_size)
        return logits


# Lord the Qwen3 base model and tokeniser
qwen_dec_model_name = "Qwen/Qwen3-0.6B-Base"
tokeniser = AutoTokenizer.from_pretrained(qwen_dec_model_name)
decoder = AutoModelForCausalLM.from_pretrained(qwen_dec_model_name)


