#script1.sh
clear
echo "Starting shell script .."
pip install tf_slim
pip install model-profiler
#pip install tensorflow
#pip install tensorflow-datasets
#pip install tfds-nightly

python3 pakkon.py
python3 M2/run.py --train_epochs=20 --m2_epoch=90 --learning_rate=0.3 --temperature=0.5

python3 pakkon.py
python3 M2/run.py --train_epochs=10 --m2_epoch=100
python3 pakkon.py
python3 M2/run.py --train_epochs=5 --m2_epoch=90
python3 pakkon.py
python3 M2/run.py --train_epochs=5 --m2_epoch=100
python3 pakkon.py
python3 M2/run.py --train_epochs=15 --m2_epoch=90
python3 pakkon.py
python3 M2/run.py --train_epochs=15 --m2_epoch=100
python3 pakkon.py
python3 M2/run.py --train_epochs=15 --m2_epoch=80
python3 pakkon.py
python3 M2/run.py --train_epochs=5 --m2_epoch=110
python3 pakkon.py
python3 M2/run.py --train_epochs=10 --m2_epoch=120

python3 pakkon.py
python3 M3/run.py --train_epochs=10 --m2_epoch=40 --m3_epoch=90
python3 pakkon.py
python3 M3/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=100
python3 pakkon.py
python3 M3/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=110
python3 pakkon.py
python3 M3/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=130
python3 pakkon.py
python3 M3/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=140
#python3 pakkon.py
#python3 M3/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=90 --learning_rate=0.5
#python3 pakkon.py
#python3 M3/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=90 --learning_rate=0.1
#python3 pakkon.py
#python3 M3/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=90 --temperature=0.3

python3 pakkon.py
python3 M1/run.py --train_epochs=100

python3 pakkon.py
python3 M4/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=30 --m4_epoch=100
python3 pakkon.py
python3 M4/run.py --train_epochs=10 --m2_epoch=20 --m3_epoch=40 --m4_epoch=100


echo "Ending of the script";

cat run.sh
