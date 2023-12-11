from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

device = "cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.add_tokens(["<bot>: "])

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2-large")
model.resize_token_embeddings(len(tokenizer))

# Move model to the chosen device
model = model.to(device)


model.load_state_dict(torch.load("model_state_2_large.pt", map_location=torch.device(device)))

# Function for model inference
def infer(model, tokenizer, device, text):
    model.eval()
    input = text
    input_ids = tokenizer(input, return_tensors="pt")['input_ids'].to(device)
    attention_mask = tokenizer(input, return_tensors="pt")['attention_mask'].to(device)
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150,
                            num_beams=5, no_repeat_ngram_size=2, do_sample=True, 
                            top_k=50, top_p=0.95, temperature=1)
        
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Flask routes
@app.route('/')
def index():
    return render_template('home2.html')

@app.route('/infer', methods=['POST'])
def chatbot_response():
    if request.method == 'POST':
        user_input = request.get_json()['user_input']
        answer = infer(model, tokenizer, device, user_input)
        answer = answer.split(user_input)[1]
        return jsonify({'bot_response': answer})
    
# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)




