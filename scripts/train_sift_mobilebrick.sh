export CUDA_VISIBLE_DEVICES=0


# for SCENE in 'aston' 'audi' 'beetles' 'big_ben' 'boat' 'bridge' 'cabin' 'convertible' 'ferrari' 'jeep' 'castle' 'london_bus' 'colosseum' 'camera'  'motorcycle' 'porsche' 'satellite' 'space_shuttle';
for SCENE in 'aston';
do
# python train.py \
# --mode train --conf confs/mobilebrick_sift_porf.conf \
# --case $SCENE

python train.py \
--mode train --conf confs/mobilebrick_sift_pose.conf \
--case $SCENE

done