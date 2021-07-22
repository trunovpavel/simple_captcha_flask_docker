docker run --rm -it -v $PWD/flask_service:/flask_service -p 5000:5000 flask_captcha /bin/bash -c 'cd ../flask_service && ./start_gunicorn.sh'
