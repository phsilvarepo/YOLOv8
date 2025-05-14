FROM ghcr.io/twocolors/node-red-docker:latest

# Install required npm packages globally
RUN npm install tfjs jimp

RUN npm install node-red-contrib-tensorflow 

COPY settings.js /data/settings.js
COPY flows.json /data/flows.json

WORKDIR /data

# Install contrib nodes into user directory (same as Palette Manager)
RUN npm install \
    node-red-contrib-tensorflow \
    node-red-contrib-browser-utils \
    node-red-contrib-image-output \
    node-red-contrib-post-object-detection \
    node-red-contrib-tf-function \
    node-red-contrib-tf-model \
    node-red-contrib-image-tools
    
WORKDIR /usr/src/node-red

COPY parts_web_model /data/parts_web_model/
COPY cases_web_model /data/cases_web_model/
