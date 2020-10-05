FROM lambci/lambda:build-python3.7

WORKDIR /tmp

# We install titiler and mangum

COPY README.md /app/README.md
COPY titiler/ /app/titiler/
COPY setup.py /app/setup.py

COPY wheels/lambda/tflite_runtime-2.4.0-cp37-cp37m-linux_x86_64.whl /app
RUN pip install /app/.["lambda"] /app/tflite_runtime-2.4.0-cp37-cp37m-linux_x86_64.whl -t /var/task --no-binary numpy,pydantic

# Reduce package size and remove useless files
RUN cd /var/task && find . -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp $f $n; done;
RUN cd /var/task && find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN cd /var/task && find . -type f -a -name '*.py' -print0 | xargs -0 rm -f
RUN find /var/task -type d -a -name 'tests' -print0 | xargs -0 rm -rf
RUN rm -rdf /var/task/numpy/doc/
RUN rm -rdf /var/task/stack

RUN cd /var/task && zip -r9q /tmp/package.zip *

COPY lambda/handler.py handler.py
RUN zip -r9q /tmp/package.zip handler.py

COPY keras-models/cloud_segmentation_20200923_1844.tflite cloud_segmentation_20200923_1844.tflite
RUN zip -r9q /tmp/package.zip cloud_segmentation_20200923_1844.tflite
