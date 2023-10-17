#!/bin/bash -x
## Create python virtual env with dependencies
cd DiApprox
virtualenv venv_DiApprox
source venv_DiApprox/bin/activate
pip3 install -r requirements.txt
