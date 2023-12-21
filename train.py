from transformers import GPT2LMHeadModel, GPT2Tokenizer
from DatasetChatbot import DatasetChatbot
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys


def train(trainData, valData, model, optim, device, path, epochs=10):
    """
Train the given model using the provided training data and validate on the validation data.

Parameters:
- trainData (DataLoader): DataLoader for the training dataset.
- valData (DataLoader): DataLoader for the validation dataset.
- model (nn.Module): The neural network model to be trained.
- optim (torch.optim.Optimizer): The optimizer for updating the model's parameters.
- device (torch.device): The device (CPU or GPU) on which the model and data should be placed.
- path (str): Path to save the trained model.
- epochs (int): Number of training epochs. Default is 10.

Returns:
- train_losses (list): List of average training losses at each epoch.
- val_losses (list): List of average validation losses at each epoch.
"""   

    # we initialize the train and validation losses to the loss of the model before training
    train_losses = []
    val_losses = []

    # as a baseline, we compute the loss of the model before training both on the training and validation data
    model.eval()
    epoch_val_loss = []
    epoch_loss = []
    for x in trainData:
        x = x.to(device)
        loss = model(x, labels=x).loss
        epoch_loss.append(loss.item())
    train_losses.append(sum(epoch_loss)/len(epoch_loss))
    for x in valData:
            x = x.to(device)
            loss = model(x, labels=x).loss
            epoch_val_loss.append(loss.item())
    val_losses.append(sum(epoch_val_loss)/len(epoch_val_loss))
    print("Initial train loss: {}".format(train_losses[-1]))
    print("Initial val loss: {}".format(val_losses[-1]))

    
    # we start the actual fine-tuning
    for i in tqdm.tqdm(range(epochs)):
        model.train() # we set the model to train mode
        epoch_loss = [] # we reset the list of losses for the current epoch
        for x in trainData:
            x = x.to(device) # we move the data to the device
            optim.zero_grad() # we reset the gradients
            loss = model(x, labels=x).loss   # we compute the loss
            epoch_loss.append(loss.item())   # we append the loss to the list of losses for the current epoch
            loss.backward() # we compute the gradients
            optim.step()   # we update the parameters
        train_losses.append(sum(epoch_loss)/len(epoch_loss))
        model.eval() # we set the model to evaluation mode in order to compute the validation loss
        epoch_val_loss = []
        for x in valData:
            x = x.to(device)
            loss = model(x, labels=x).loss
            epoch_val_loss.append(loss.item())
        val_losses.append(sum(epoch_val_loss)/len(epoch_val_loss))
        # we save the model only if it produced the lowest validation loss
        if(val_losses[-1] == min(val_losses)):
            torch.save(model.state_dict(), path + "/model_state_2_large_v3.pt")

        print("Epoch: {}, Train loss: {}, Val loss: {}".format(i, train_losses[-1], val_losses[-1]))

    
    return train_losses, val_losses

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 train.py <path_to_save_model>")
        sys.exit(1)


    device = "cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("Loading model...")
    # load the model and tokenizer, both pretrained on gpt2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.add_tokens(["<bot>: "])

    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    print("Model loaded!")

    # load the dataset and split it into training, validation and test sets
    data_path = "data/qa_heatpumps_2-2.csv"
    chatData = DatasetChatbot(data_path, tokenizer)
    trainData, testData = train_test_split(chatData, test_size=0.4, random_state=42)
    valData, testData = train_test_split(testData, test_size=0.2, random_state=42)

    # create the dataloaders in order to train the model
    trainLoader = DataLoader(trainData, batch_size=16, shuffle=True)
    valLoader = DataLoader(valData, batch_size=16, shuffle=True)
    testLoader = DataLoader(testData, batch_size=16, shuffle=True)

    # create the optimizer, we use Adam with a learning rate of 1e-5
    optim = Adam(model.parameters(), lr=1e-5)

    print("Starting training...\n")
    train_losses, val_losses = train(trainLoader, valLoader, model, optim, device, sys.argv[1], epochs=5)
