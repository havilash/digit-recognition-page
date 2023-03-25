FROM tensorflow/tensorflow
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
# EXPOSE 5000
ENTRYPOINT [ "python" ] 
CMD [ "main.py" ]