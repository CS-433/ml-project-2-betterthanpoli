from torch.utils.data import Dataset
import pandas as pd

class DatasetChatbot(Dataset):
    def __init__(self, file_path, tokenizer):
        #Read csv file with question and answer columns
        self.data = pd.read_csv(file_path, sep=';')
        # Initialize lists to hold question and answer
        self.X = []
        self.Q = []
        self.A = []
        for i in range(len(self.data)):
            #Add question to Q
            self.Q.append(self.data.iloc[i]['Question'])
            #Add answer to A
            self.A.append(self.data.iloc[i]['Answer'])
            self.tokenizer = tokenizer
        for i in range(len(self.Q)):
            full_text = self.Q[i] + " " + self.A[i] + " <|endoftext|>"  #Add the string " <|endoftext|>" in order to make the tokenizer output the EOS Token
            self.X.append(full_text)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # We will return the econded input 
        input_ids = self.tokenizer.encode(self.X[index], max_length=100, truncation=True, padding='max_length', return_tensors="pt").squeeze()
        return input_ids
    
    
