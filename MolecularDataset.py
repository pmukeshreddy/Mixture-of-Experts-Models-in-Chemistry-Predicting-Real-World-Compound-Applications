class MolecularDataset(Dataset):
    def __init__(self,root,filename,smiles_list,labels_list,transform,pre_transform):
        self.smiles_list = smiles_list
        self.labels_list = labels_list
        super(MolecularDataset,self).__init__(root,transform,pre_transform)
        if not self.data_list:
            self.process()
    def process(self):
        self.data_list = []
        for i , smile in enumerate(self.smiles_list):
            label = self.labels_list[i]
            graph_data = smiles_to_graph_data(smile,label)
            if graph_data is not None:
                self.data_list.append(graph_data)
    def len(self):
        return len(self.data_list)
    def get(self,idx):
        return self.data_list[idx]
