from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/upload', methods=['POST'])
def upload_csv(request):
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('csv'):
            return "File uploaded successfully!"

    return "Invalid file format. Please upload a CSV file."



# Flow -> upload csv -> analyze csv data -> visualize data -> train data