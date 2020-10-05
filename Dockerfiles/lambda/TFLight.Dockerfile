# https://www.edeltech.ch/tensorflow/machine-learning/serverless/2020/07/11/how-to-deploy-a-tensorflow-lite-model-on-aws-lambda.html

FROM lambci/lambda:build-python3.7

WORKDIR /tmp

RUN pip install numpy pybind11
RUN git clone https://github.com/tensorflow/tensorflow.git
RUN cd tensorflow && sh tensorflow/lite/tools/pip_package/build_pip_package.sh

# at this point I had to comment out a line from a file to avoid an error
# the code was an assert