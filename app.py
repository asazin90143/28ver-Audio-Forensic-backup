from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Audio Forensic Analysis Service Active"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
