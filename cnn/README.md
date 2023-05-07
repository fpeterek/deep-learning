python3 cnn/train.py train-resnet --model-name 'resnet.model'
python3 cnn/train.py train-small --model-name 'small-lr001-b64.model' --dataset data/96/
python3 cnn/train.py train-hog --model-name 'svm.model'
python3 cnn/train.py train-small --model-name 'small-lr001-b16.model' --lr 0.001 --batch-size 16 --dataset data/96/
python3 cnn/train.py train-small --model-name 'small-lr003-b64.model' --lr 0.003 --dataset data/96/
python3 cnn/train.py train-small --model-name 'small-enhanced.model' --lr 0.001 --batch-size 32 --dataset data/enhanced
python3 cnn/train.py train-large --model-name 'large-enhanced.model' --lr 0.001 --batch-size 32 --dataset data/enhanced
python3 cnn/train.py train-large --model-name 'large-lr001-b8.model' --lr 0.001 --batch-size 8 --dataset data/96
python3 cnn/train.py train-large --model-name 'large-lr01-b64.model' --lr 0.01 --batch-size 64 --dataset data/96
python3 cnn/train.py train-large --model-name 'large-lr001-b64.model' --lr 0.001 --batch-size 64 --dataset data/96
