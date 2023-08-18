FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt update
RUN apt install -y git python3 python3-pip wget
WORKDIR /all
COPY inpainting.py /all/
COPY text_to_image.py /all/
COPY img_to_img.py /all/
COPY requirements.txt /all/
RUN pip install scipy
RUN pip install -r requirements.txt
CMD ["python","text_to_image.py","img_to_img.py","inpainting.py"]
