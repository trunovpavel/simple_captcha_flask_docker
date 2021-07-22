FROM pytorch/pytorch
ENV DEBIAN_FRONTEND=noninteractive   
COPY ./flask_service/requirements.txt ./
RUN apt-get -y update \
	&& apt-get install -y vim \
	&& apt-get install -y python3-opencv \	
	&& pip3 install -r requirements.txt \
	&& pip3 install gunicorn
