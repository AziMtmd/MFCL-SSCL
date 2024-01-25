#script1.sh
clear
echo "Starting shell script .."
pip install tf_slim
pip install model-profiler
#pip install tensorflow
#pip install tensorflow-datasets
#pip install tfds-nightly

python3 pakkon.py
python3 M4/run.py 

echo "Ending of the script";

cat run.sh
