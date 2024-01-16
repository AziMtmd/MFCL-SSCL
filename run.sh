#script1.sh
clear
echo "Starting shell script .."
pip install tf_slim
pip install model-profiler
#pip install tensorflow
#pip install tensorflow-datasets
#pip install tfds-nightly


python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=0.6 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=0.6 --temperature=0.4
python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=0.5 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=0.5 --temperature=0.4
python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=0.7 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=0.8 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=1 --temperature=0.3
python3 pakkon.py
python3 M1/run.py --train_epochs=100 --learning_rate=1.5 --temperature=0.3


python3 pakkon.py
python3 M2/run.py --train_epochs=100 --m2_epoch=5 --learning_rate=0.5 --temperature=0.3
python3 pakkon.py
python3 M2/run.py --train_epochs=100 --m2_epoch=15 --learning_rate=0.7 --temperature=0.3
python3 pakkon.py
python3 M2/run.py --train_epochs=100 --m2_epoch=15 --learning_rate=0.8 --temperature=0.3
python3 pakkon.py
python3 M2/run.py --train_epochs=100 --m2_epoch=15 --learning_rate=1 --temperature=0.3
python3 pakkon.py
python3 M2/run.py --train_epochs=100 --m2_epoch=15 --learning_rate=1.5 --temperature=0.3



#python3 pakkon.py
#python3 M3/run.py --train_epochs=100 --learning_rate=0.6 --temperature=0.3
#python3 pakkon.py
#python3 M3/run.py --train_epochs=100 --learning_rate=0.6 --temperature=0.4
#python3 pakkon.py
#python3 M3/run.py --train_epochs=100 --learning_rate=0.5 --temperature=0.3
#python3 pakkon.py
#python3 M3/run.py --train_epochs=100 --learning_rate=0.5 --temperature=0.4



echo "Ending of the script";

cat run.sh
