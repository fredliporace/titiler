FROM lambci/lambda:build-python3.7

WORKDIR /tmp

RUN pip install titiler==0.1.0a14 mangum>=0.10.0 -t /var/task --no-binary numpy,pydantic

# Hack to avoid stac_version not found in pystac
RUN cd /var/task && sed -i "s/d.pop('stac_version')/d.pop('stac_version', None)/g" pystac/item.py
RUN cd /var/task && python -m compileall pystac/item.py

# Reduce package size and remove useless files
RUN cd /var/task && find . -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp $f $n; done;
RUN cd /var/task && find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN cd /var/task && find . -type f -a -name '*.py' -print0 | xargs -0 rm -f
RUN find /var/task -type d -a -name 'tests' -print0 | xargs -0 rm -rf
RUN rm -rdf /var/task/numpy/doc/
RUN rm -rdf /var/task/uvicorn

COPY lambda/handler.py /var/task/handler.py
