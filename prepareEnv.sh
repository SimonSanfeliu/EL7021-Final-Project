git clone https://github.com/google-research/language-table.git
#mv language-table language_table
cd language_table
python3 -m venv ./ltvenv
source ./ltvenv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=${PWD}:$PYTHONPATH