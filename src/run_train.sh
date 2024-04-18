# Get the absolute path of the directory containing the script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd $SCRIPT_DIR

python3 -m run.train_hyperprior -e 150 -n 32 --batch-size 352 

cd -
