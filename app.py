from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import bedo  # Import your model

app = Flask(__name__)
model = bedo()

# Upload folder for PDF (if using RAG model)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Check for allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for rendering the form
@app.route('/')
def index():
    return render_template('index.html')  # Renders your HTML form

# Route for processing form submission
@app.route('/query', methods=['POST'])
def query():
    model_type = request.form.get('model_type')  # Get model type from the form
    question = request.form.get('question')  # Get the question from the form
    url = request.form.get('url')  # Get the URL if it's provided (for image model)
    pdf_file = request.files.get('pdf')  # Get PDF file if uploaded (for RAG model)

    # Handle file upload for PDF
    pdf_path = None
    if pdf_file and allowed_file(pdf_file.filename):
        filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(pdf_path)

    # Validation: Ensure that question is provided when necessary
    if not question and model_type in ['generative', 'rag']:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Process the data based on model type
        if model_type == 'image' and url:
            answer = model(image_url=url)  # Handle image model
        else:
            answer = model(question=question, url=url, pdf_path=pdf_path, model_type=model_type)

        return jsonify({"answer": answer})  # Respond with the answer

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5400)
