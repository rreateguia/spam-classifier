import torch 
import nltk
import regex as re
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from collections import Counter

def data_extraction(train_url, test_url):
    #Read data from urls using parquet
    splits = {'train': train_url, 'test': test_url}
    df_train = pd.read_parquet("hf://datasets/Deysi/spam-detection-dataset/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/Deysi/spam-detection-dataset/" + splits["test"])
    return [df_train, df_test]

#Data cleaning

def clean_text(text):

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))#set of stopwords in a language
    text = text.lower()#lowercase all words
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)#remove alphanumeric characters
    words = [w for w in text.split() if w not in stop_words]#create new set of words
    return " ".join(words)

#stopwords: eliminate words that do not provide semantical meaning
#nltk library will handle stopwords
def data_cleaning(df_train, df_test):

    #apply the cleaning to the text column and store the result in the clean_text column
    df_train['clean_text'] = df_train['text'].apply(clean_text)
    df_test['clean_text'] = df_test['text'].apply(clean_text)

    #Map Spam->1 Not Spam->0
    df_train["label"] = df_train["label"].map({"spam": 1, "not_spam": 0}).astype("float32")
    df_test["label"] = df_test["label"].map({"spam": 1, "not_spam": 0}).astype("float32")
    return [df_train, df_test]

#Tokenization and vocabulary

#<PAD>  (padding token): Used to truncate sequences to a uniform length, 
# ensuring that all input sequences have the same length, which is necessary for batch processing.
#<UNK>  (unknown token): Used to represent out-of-vocabulary (OOV) words, 
# that is, words that are not in the training dataset, allowing the model 
# to handle unseen words during inference.
def tokenize(text):#separate into tokens
    return text.split()

def encode(text, vocab, max_len=50):#encoded text will have length 50 
    tokens = tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens] #get the corresponding ids of every text 
    if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))#padding if it has length < 50 
    else:
            ids = ids[:max_len]#cut off if it has more than 50 ids
    return ids #return ids of texts

#Define class that handles the encoded data(texts represented by ids and labels as tensors)
class SpamDataset(Dataset):

        def __init__(self, texts, labels, vocab):
            self.texts = [torch.tensor(encode(t, vocab, max_len=50), dtype=torch.long) for t in texts] #converts the texts into tensors with id of each token
            self.labels = torch.tensor(labels, dtype=torch.float32) #one tensor for all the labels

        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, index):
            return self.texts[index], self.labels[index]
        
def data_prep(train_url, test_url):

    [df_train, df_test] = data_extraction(train_url, test_url)
    [df_train, df_test] = data_cleaning(df_train, df_test)
    
    train_texts = df_train["clean_text"]

    all_tokens = [token for text in train_texts for token in tokenize(text)]#get all words(tokens) for all the texts

    vocab = {word: i+2 for i, word in enumerate(Counter(all_tokens))} 
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1 #vocab: enumerate all tokens    
               
    #Define instances
    train_texts = df_train['clean_text']
    train_labels = df_train['label'].to_numpy(copy=True)
    
    test_texts = df_test['clean_text']
    test_labels = df_test['label'].to_numpy(copy=True)

    train_ds = SpamDataset(train_texts, train_labels, vocab)
    test_ds = SpamDataset(test_texts, test_labels, vocab)

    return [train_ds, test_ds, vocab]