~~~
docker build -t hello:v1
docker run  hello:v1
docker run --it -rm hello:v1 bash
docker run --it -rm --pid=host hello:v1 bash
docker run --it -rm -m 512mb hello:v1 bash
~~~