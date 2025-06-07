class MoELayer(nn.Module):
    def __init__(self,input_dim,output_dim,num_experts,expert_hidden_dim,top_k,load_balance_loss_coeff,dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_loss_coeff = load_balance_loss_coeff

        self.experts = nn.ModuleList([
          Expert(input_dim,expert_hidden_dim,output_dim,dropout) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim,num_experts)

    def forward(self,x):
        """
        Args:
            x : Input tensor , shape [batch_size,seq_len,input_dim]
        Returns:
            final_output : Tensor of shape [batch_size,seq_len,output_dim]
            load_blacking : a scalar for auxiliary loss
        """
        batch_size,seq_len , _ = x.shape
        x_flat = x.reshape(-1,self.input_dim) #[batch_size*seq_len,input_dim]

        router_logits = self.gate(x_flat) #[num_tokens,num_epxerts]
        gatting_probs = F.softmax(router_logits,dim=-1)

        top_k_gatting_weights , top_k_indices = torch.topk(gatting_probs,dim=-1)

        top_k_gatting_weights_norm = top_k_gatting_weights / (top_k_gatting_weights.sum(dim=-1,keepdim=True)+1e-6)

        final_output_flat = torch.zeros(x_flat)

        expert_mask_flat = F.one_hot(top_k_indices,num_classes=self.num_experts).float()

        tokens_per_indicator = expert_mask_flat.sum(dim=1)

        f_i = tokens_per_indicator.mean(dim=0)

        p_i = gatting_probs.mean(dim=0)

        load_balancing_loss = self.num_experts * torch.sum(f_i*p_i)

        for k_choice_idx in range(self.top_k):
            current_expert_indicies_for_choice_k = top_k_indices[:,k_choice_idx]
            current_gatting_weights_for_choice_k = top_k_gatting_weights_norm[:,k_choice_idx]

            for expert_id in range(self.num_experts):
                expert_module = self.experts[expert_id]

                token_indices_for_this_expert_at_k = torch.where(current_expert_indicies_for_choice_k == expert_id)

                if token_indices_for_this_expert_at_k.numel() > 0:
                    selected_tokens_input = x_flat[token_indices_for_this_expert_at_k]

                    expert_output = expert_module(selected_tokens_input)

                    weighted_expert_output = expert_output * current_gatting_weights_for_choice_k[token_indices_for_this_expert_at_k].unsqueeze(1)

                    final_output_flat.index_add_(0,token_indices_for_this_expert_at_k,weighted_expert_output)
                    
        final_output = final_output_flat.reshape(batch_size,seq_len,self.output_dim)
        return final_output , self.load_balance_loss_coeff * load_balancing_loss
                    
            
