FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get upgrade -y && \
    apt-get install build-essential -y

COPY . /project

WORKDIR /project

RUN pip install -e .

EXPOSE 8080

CMD ["uvicorn", "mediqa.app:app", "--host=0.0.0.0", "--port=8080"]