source ./ltvenvtrain/bin/activate
mkdir -p /tmp/language_table_train/
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./language_table/language_table/train/main.py \
    --config=./language_table/train/configs/language_table_sim_local.py \
    --workdir=/tmp/language_table_train/ \
    --dataset_path=./spanish_obstacle_dataset.json