REG=4e-4
echo $REG

python train.py --lr=1e-3 --delete=True --n_epochs=10 --reg=$REG
python train.py --lr=1e-4 --n_epochs=5 --reg=$REG

python evaluate.py --data=train
python evaluate.py --data=cv
