FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
COPY . /home/
WORKDIR /home/
RUN pip install -r requirements.txt && pip install --pre pytorch-ignite
