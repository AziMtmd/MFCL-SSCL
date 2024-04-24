#script1.sh
#/azi/downloads/manual. Create it and download/extract dataset artifacts in there using instructions.
#manual_dir should contain `ILSVRC2012_img_val.tar` file.
clear
echo "Starting shell script .."
pip install tf_slim
pip install model-profiler
#pip install tensorflow
#pip install tensorflow-datasets
#pip install tfds-nightly
pip install matplotlib
pip install scikit-learn


#python3 pakkon.py
#python3 M1/run.py --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=100
#python3 pakkon.py
#python3 M1/run.py --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=200

#python3 pakkon.py
#python3 M4SemiParallel/run.py --dataset='cifar10' --image_size=32 --eval_split='test' 

python3 pakkon.py
python3 M1MR/run.py --dataset='cifar10' --image_size=32 --eval_split='test' 
python3 pakkon.py
python3 M1CR/run.py --dataset='cifar10' --image_size=32 --eval_split='test' 
python3 pakkon.py
python3 M1mahd/run.py --dataset='cifar10' --image_size=32 --eval_split='test' 

#python3 pakkon.py
#python3 M1/run.py --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=100 --resnet_depth=50
#python3 M1/run.py --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=200 --resnet_depth=50
#python3 M1/run.py --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=300 --resnet_depth=50
#python3 M1/run.py --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=400 --resnet_depth=50



#python3 pakkon.py
#python3 M4AllCont/run.py --resnet_depth=18 --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=200 --m2_epoch=200 --m3_epoch=200 --m4_epoch=200

#python3 pakkon.py
#python3 M4AllCont/run.py --resnet_depth=18 --dataset='cifar10' --image_size=32 --eval_split='test' --train_epochs=200 --m2_epoch=200 --m3_epoch=200 --m4_epoch=250



#python3 pakkon.py
#python3 M1/run.py --dataset='imagenet_resized/64x64' --image_size=64 --eval_split='validation'

#python3 MF/run.py --color_jitter_strength=0.5 --train_batch_size=320 --train_epochs=400 --learning_rate=1.5 --weight_decay=1e-4 --temperature=0.3 --dataset=cifar10 --image_size=32 --eval_split=test --use_blur=False --resnet_depth=18 --model_dir=/tmp/simclr_test --tekrar=500 --NumofWorkers=10  --epochsm=1 --serBatch=32 --AutoBatch=64 --testBatch=320 --norm=GN --CPC=10


echo "Ending of the script";

cat run.sh
