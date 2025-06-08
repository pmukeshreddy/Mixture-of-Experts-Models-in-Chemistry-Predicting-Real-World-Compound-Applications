class MoETransformerforChemApp(nn.Module):
    def __init__(self,vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int,
                 moe_num_experts: int, moe_expert_hidden_dim: int, moe_top_k: int,
                 num_labels: int, # Number of chemical applications (output classes)
                 dropout_rate: float = 0.1, max_seq_len: int = 256,
                 moe_load_balance_coeff: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embeddings(vocab_size,d_model)
        self.pos_encoder = PostionalEncoding(d_model,dropout_rate,max_seq_len)

        encoder_layer_list = [MoETransformerEncoderLayer(d_model,nhead,moe_num_experts,moe_expert_hidden_dim,moe_top_k,dropout_rate,1e-5,moe_load_balance_coeff)
                             for _ in range(num_encoder_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layer_list)
        self.classification_head = nn.Linear(d_model,num_labels)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        if hasattr(self,"embeddings"):
            self.embeddings.weight.data.uniform_(-initrange,initrange)
        self.classification_head.bias.data.zero_()
        self.classification_head.bias.data.uniform_(-initrange,initrange)

    def forward(self,src_tokens_or_embeddings,padding_mask,is_embeeded):
        if is_embeeded:
            embeded_src = src_tokens_or_embeddings
        else:
            embeded_src = self.embeddings(src_tokens_or_embeddings) * math.sqrt(self.d_model)
        src = self.pos_encoder(embeded_src)
        total_aux_loss = torch.tensor(0.0,device=src.device)
        for layer in self.transformer_encoder:
            src , layer_aux_loss = layer(src,src_key_padding_mask=padding_mask)
            total_aux_loss += layer_aux_loss
        logits = self.classification_head(src[:,0,:])
        return logits , total_aux_loss
