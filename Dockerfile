FROM python:3.9

WORKDIR /code

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y ffmpeg

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Hugging Face Spaces requires port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
