appcsv3 converts a collection of image into their colored counterparts with detection boxes.
webgpt3.csv is the flask python app.
image.py has been modified to work on both RGB and monochrome images.
run the flask app via gunicorn --timeout 3600 -w 4 -b 0.0.0.0:5000 webgpt3:app (large images may have long processing times)
put the index.html file in the /templates directory.
create an imageviewer folder in the /static directory
