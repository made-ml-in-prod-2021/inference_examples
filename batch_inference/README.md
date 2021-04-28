~~~
docker build -t mikhailmar/batch_inference:v1 
docker push mikhailmar/batch_inference:v1

docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} mikhailmar/batch_inference:v1
~~~