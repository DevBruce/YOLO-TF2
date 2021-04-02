FROM tensorflow/tensorflow:2.4.1-gpu


RUN apt-get update
RUN apt-get install -y vim

# Requirement of opencv-python
RUN apt-get install -y libgl1-mesa-glx

# Install python packages
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install tensorflow-datasets
RUN pip install albumentations
RUN pip install pascalvoc-ap
RUN pip install jupyterlab
COPY . /YOLOv1_TF2
WORKDIR /YOLOv1_TF2
