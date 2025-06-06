import re

class GenricSmilesTokenizer:
    def __init__(self,smiles_corpus:Optional[List[str]]=None):
        self.SMILES_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.SMILES_REGEX_PATTERN)

        if smiles_corpus:
            all_tokens = [token for smiles in smiles_corpus for token in self.regex.findall(smiles)]
            vocab = sorted(list(set(all_tokens)))
        else:
            vocab = [
                'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', '(', ')', '[', ']',
                '=', '#', '.', 'c', 'n', 'o', 's', '1', '2', '3', '4', '5', '6',
                '7', '8', '9', '0', '-', '+', ':'
            ]
        self.special_tokens =  ['<pad>', '<unk>', '<bos>', '<eos>']
        self.vocab = self.special_tokens + vocab
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        # Set IDs for special tokens
        self.pad_token_id = self.token_to_id['<pad>']
        self.unk_token_id = self.token_to_id['<unk>']
        self.bos_token_id = self.token_to_id['<bos>']
        self.eos_token_id = self.token_to_id['<eos>']
    def _tokenize_string(self,smiles_string:str)->List[str]:
        return self.regex.findall(smiles_string)
    def tokenize(self,smiles_string:str,add_special_string:bool=True)->List[int]:
        tokens_str = self._tokenize_string(smiles_string)
        token_ids = [self.token_to_id.get(token,self.unk_token_id) for token in tokens_str]
        if add_special_string:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids
    def detokenize(self,token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in token_ids:
            token_str = self.id_to_token(token_id,"")
            if skip_special_tokens and token_str in self.special_tokens:
                continue
            tokens.append(token_str)
        return "".join(tokens)
    
