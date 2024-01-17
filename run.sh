#script1.sh
clear
echo "Starting shell script .."
pip install tf_slim
pip install model-profiler
#pip install tensorflow
#pip install tensorflow-datasets
#pip install tfds-nightly


python3 pakkon.py
python3 M1/run.py --train_epochs=25 --m2_epoch=100 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=30 --m2_epoch=100 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=35 --m2_epoch=100 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=40 --m2_epoch=100 --learning_rate=1.5 --temperature=0.3

python3 pakkon.py
python3 M1/run.py --train_epochs=25 --m2_epoch=105 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=30 --m2_epoch=110 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=35 --m2_epoch=115 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=40 --m2_epoch=115 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=25 --m2_epoch=110 --learning_rate=1.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=30 --m2_epoch=105 --learning_rate=1.5 --temperature=0.3




echo "Ending of the script";

cat run.sh
