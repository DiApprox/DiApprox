#!/bin/bash -x



## Lunch tests
cd DiApprox
# It doesn't harm to re-activiate the env just after its creation
source venv_DiApprox/bin/activate

python3 src/test_dataset_.py

