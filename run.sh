#script1.sh
clear
echo "Starting shell script .."
pip install tf_slim
#pip install tensorflow
#pip install tensorflow-datasets
#pip install tfds-nightly
python3 M2/run.py --train_epochs=10 --m2_epoch=100
python3 pakkon.py
python3 M2/run.py --train_epochs=10 --m2_epoch=80
python3 pakkon.py
python3 M2/run.py --train_epochs=5 --m2_epoch=90
python3 pakkon.py
python3 M2/run.py --train_epochs=5 --m2_epoch=100
python3 pakkon.py
python3 M2/run.py --train_epochs=5 --m2_epoch=80
python3 pakkon.py
python3 M2/run.py --train_epochs=15 --m2_epoch=90
python3 pakkon.py
python3 M2/run.py --train_epochs=15 --m2_epoch=100
python3 pakkon.py
python3 M2/run.py --train_epochs=15 --m2_epoch=80

python3 M3/run.py --train_epochs=10 --m2_epoch=100
python3 pakkon.py
python3 M3/run.py --train_epochs=10 --m2_epoch=80
python3 pakkon.py
python3 M3/run.py --train_epochs=5 --m2_epoch=90
python3 pakkon.py
python3 M3/run.py --train_epochs=5 --m2_epoch=100
python3 pakkon.py
python3 M3/run.py --train_epochs=5 --m2_epoch=80
python3 pakkon.py
python3 M3/run.py --train_epochs=15 --m2_epoch=90
python3 pakkon.py
python3 M3/run.py --train_epochs=15 --m2_epoch=100
python3 pakkon.py
python3 M3/run.py --train_epochs=15 --m2_epoch=80

echo "Ending of the script";

cat run.sh
