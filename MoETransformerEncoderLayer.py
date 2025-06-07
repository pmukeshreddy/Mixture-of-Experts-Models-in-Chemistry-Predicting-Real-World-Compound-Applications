class MoETransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,nhead,moe_num_experts,moe_expert_hidden_dim,moe_top_k,dropout,
                layer_norma_eps,moe_load_balance_coef):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=True)
        self.moe_layer = MoELayer(input_dim=d_model,output_dim=d_model,num_experts=moe_num_experts,expert_hidden_dim=moe_expert_hidden_dim,
                                 top_k=moe_top_k,load_balance_loss_coeff=moe_load_balance_coef,dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model,eps=layer_norma_eps)
        self.norm2 = nn.LayerNorm(d_model,eps=layer_norma_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self,src,src_mask,src_key_padding_mask):
        attn_output = self.self_attn(src,src,src,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=False)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        moe_output , aux_loss = self.moe_layer(src)
        src = src + self.dropout2(moe_output)
        src = self.norm2(src)
        return src,aux_loss
        
