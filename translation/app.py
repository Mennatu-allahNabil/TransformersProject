from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load local model
custom_model_path = "../Models/first model"
tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(custom_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        input_text = request.form['text']
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(**inputs)
        predicted_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return render_template('index.html', result={'translation': predicted_translation}, text=input_text)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)