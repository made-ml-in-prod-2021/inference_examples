~~~
docker build -t mikhailmar/online_inference:v1 .
docker run -p 8000:8000 mikhailmar/online_inference:v1
python make_request.py 

~~~