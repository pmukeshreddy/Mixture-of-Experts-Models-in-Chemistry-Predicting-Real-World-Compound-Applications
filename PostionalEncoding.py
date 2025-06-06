class PostionalEncoding(nn.Module):
    def __init__(self,d_model:int,dropout:float=0.1,max_len:int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arrange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe = torch.zeroes(max_len,1,d_model)
        pe[:,0,0::2] = torch.sin(position*div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)
        self.pe = pe
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        x = x + self.pe[:x.size(1),0,:].unsqueeze(0)
        return self.dropout(x)
