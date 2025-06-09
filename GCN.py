class GCN(nn.Module):
    def __init__(self,num_node_features,num_gnn_features,hidden_channels,gnn_output_dim,drop_out):
        super(GCN,self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features,hidden_channels))
        for _ in range(num_gnn_features-1):
            self.convs.append(GCNConv(hidden_channels,hidden_channels))
        self.dropout = nn.Dropout(p=drop_out)
        self.fc = nn.Linear(hidden_channels,gnn_output_dim)
    def forward(self,x):
        x , edge_index , batch = data.x , data.edge_index , data.batch
        for conv in self.convs:
            x = conv(x , edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x_pooled = global_mean_pool(x,batch)
        graph_embeddings = self.fc(x_pooled)
        return graph_embeddings
