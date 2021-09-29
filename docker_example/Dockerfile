FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# copy files
ADD scripts /workspace/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
RUN pip install nibabel
RUN pip install Pillow
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
