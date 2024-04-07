
# Get the absolute path of the directory containing the script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd $SCRIPT_DIR

python3 -m run.test_hyperprior --batch-size 188 --checkpoint checkpoint_best_loss_0.0018.pth.tar

cd -
