class GNN_MoE_Model(nn.Module):
    def __init__(self,gnn,moe_transformer,gnn_output_dim,transformer_d_model):
        super(GNN_MoE_Model,self).__init()
        self.gnn = gnn
        self.moe_transformer = moe_transformer
        if gnn_output_dim != transformer_d_model:
            self.projection = nn.Linear(gnn_output_dim,transformer_d_model)
        else:
            self.projection = nn.Identity()
            
    def forward(self,graph_data):
        graph_embeddings = self.gnn(graph_data) #[batch_size,gnn_outputsize]
        projected_embeddings = self.projection(graph_embeddings) #[batch_size,transformer_d_model]
        transformer_input_embeddings = projected_embeddings.unsqueeze(1) #[batch_size,1,transformer_d_model]
        batch_size = transformer_input_embeddings.size(0)
        device = transformer_input_embeddings.device
        transformer_padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        logits , total_aux_loss = self.moe_transformer(src_tokens_or_embeddings=transformer_input_embeddings,padding_mask=transformer_padding_mask,
                                                      is_embedded=True)
        
        return logits , total_aux_loss
