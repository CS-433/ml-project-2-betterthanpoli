from transformers import GPT2LMHeadModel, GPT2Tokenizer
from DatasetChatbot import DatasetChatbot
import torch
from sklearn.model_selection import train_test_split

# We define the device and load the model, in its not fine-tuned version. We also load the tokenizer
device = "cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.add_tokens(["<bot>: "])

model = GPT2LMHeadModel.from_pretrained("gpt2-large")
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

# We load the dataset and split it into train, test and validation sets
path = "qa_heatpumps_2.csv"
chatData = DatasetChatbot(path, tokenizer)
trainData, testData = train_test_split(chatData, test_size=0.4, random_state=42)
valData, testData = train_test_split(testData, test_size=0.2, random_state=42)

test_losses = []
model.eval()
for x in testData:
    x = x.to(device)
    loss = model(x, labels=x).loss
    test_losses.append(loss.item())

avg_test_loss_no_finetune = sum(test_losses)/len(test_losses)
# Print the results
print("Average Test loss without finetuning: ", avg_test_loss_no_finetune)

# We load the fine-tuned model and perform tha same evaluation on the test set
model.load_state_dict(torch.load("model_state_2_large_v2.pt"))

test_losses = []
model.eval()
for x in testData:
    x = x.to(device)
    loss = model(x, labels=x).loss
    test_losses.append(loss.item())

avg_test_loss_finetune = sum(test_losses)/len(test_losses)

# Print the results
print("Average Test loss with finetuning: ", avg_test_loss_finetune)
