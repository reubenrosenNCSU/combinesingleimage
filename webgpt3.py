import os
import zipfile
import subprocess
from flask import Flask, request, send_file, render_template, url_for
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)

# Define folder paths
UPLOAD_FOLDER = 'uploads'
IMAGES_FOLDER = 'images'
OUTPUT_FOLDER = 'output'
FINALOUTPUT_FOLDER = 'finaloutput'
INPUT_FOLDER = 'input'
SAFE_DIRECTORY = '/home/greenbaum-gpu/Reuben/keras-retinanet/'
IMAGEVIEWER_FOLDER = os.path.join(app.static_folder, 'imageviewer')  # Change to the static folder

# Allowed file types
ALLOWED_EXTENSIONS = {'tiff', 'tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file is a valid type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder_path):
    """Helper function to clear all files in a folder, 
    but keep the directory structure intact."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Helper function to run subprocesses in order
def run_subprocess(script, args):
    try:
        subprocess.run([script] + args, check=True, timeout=600)  # Set an appropriate timeout
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {script} with args {args}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")

# Handle the upload of the image
@app.route('/')
def index():
    return render_template('index.html', download_ready=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Sequentially run subprocesses with proper exception handling
        run_subprocess('python3', ['normalize2.py'])
        run_subprocess('python3', ['splitimagev2.py', filepath])
        run_subprocess('python3', ['appcsv3.py'])
        run_subprocess('python3', ['mergeimage.py'])
        run_subprocess('python3', ['readingcsv2.py'])

        # Create the finaloutput.zip file in the specified directory
        zip_filename = 'finaloutput.zip'  # Always the same name
        zip_filepath = os.path.join(SAFE_DIRECTORY, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for root, dirs, files in os.walk(FINALOUTPUT_FOLDER):
                for file in files:
                    if file == zip_filename:
                        continue
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), FINALOUTPUT_FOLDER))

        # Copy the detections.png to the imageviewer directory (before clearing)
        detections_image = os.path.join(FINALOUTPUT_FOLDER, 'detections.png')
        if os.path.exists(detections_image):
            shutil.copy(detections_image, IMAGEVIEWER_FOLDER)

        # Clear the directories after processing
        clear_folder(UPLOAD_FOLDER)
        clear_folder(IMAGES_FOLDER)
        clear_folder(OUTPUT_FOLDER)
        clear_folder(INPUT_FOLDER)
        clear_folder(FINALOUTPUT_FOLDER)

        # Provide a download link on the index page
        return render_template('index.html', download_ready=True, zip_filename=zip_filename)

    return 'Invalid file format', 400

@app.route('/download/<filename>')
def download_file(filename):
    zip_filepath = os.path.join(SAFE_DIRECTORY, filename)

    return send_file(zip_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)  # Use threaded mode to handle multiple requests
