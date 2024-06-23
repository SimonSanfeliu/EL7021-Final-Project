python3 -m venv ./ltvenvtrain
source ./ltvenvtrain/bin/activate
pip install --no-deps -r requirements_static.txt
export PYTHONPATH=${PWD}:$PYTHONPATH