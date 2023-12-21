from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# we load the model and tokenizer, both pretrained on gpt2, putting the model on the GPU if available
device = "cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.add_tokens(["<bot>: "])

model = GPT2LMHeadModel.from_pretrained("gpt2-large")
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

# we load the fine-tuned model
model.load_state_dict(torch.load("model_state_2_large_v2.pt"))

def infer(model, tokenizer, device, text):
    """
    This function performs inference using a given version of a model and its tokenzier.

    Parameters:
    - model(nn.Module): the model to use for inference
    - tokenizer(PreTrainedTokenizer): the tokenizer to use to encode the input text and decode the output text
    - device(str): the device to use for inference
    - text(str): the text to use as input for the model
    """
    model.eval()
    input = text
    input_ids = tokenizer(input, return_tensors="pt")['input_ids'].to(device)
    attention_mask = tokenizer(input, return_tensors="pt")['attention_mask'].to(device)
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150)
        
    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.route('/')
def index():
    return render_template('home2.html')

@app.route('/infer', methods=['POST'])
def chatbot_response():
    """
    This function is the endpoint for the chatbot. It receives a POST request with a JSON object containing the user input and returns a JSON object containing the chatbot response.
    """
    if request.method == 'POST':
        user_input = request.get_json()['user_input']
        # check the presence of the question mark
        if user_input[-1] != "?":
            user_input += "?"
        answer = infer(model, tokenizer, device, user_input)
        answer = answer.split(user_input)[1] # we remove the user input from the answer since the model start from the input repeating it
        return jsonify({'bot_response': answer})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)





