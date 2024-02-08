export CUDA_VISIBLE_DEVICES=0

# for SCENE in 'scan37' 'scan65' 'scan69' 'scan83' 'scan97' 'scan105' 'scan110' 'scan118' 'scan106'  'scan114'  'scan122' 'scan24' 'scan40'  'scan55' 'scan63';

for SCENE in 'scan37';
do
python train.py \
--mode train --conf confs/dtu_sift_porf.conf \
--case $SCENE

# python train.py \
# --mode train --conf confs/dtu_sift_pose.conf \
# --case $SCENE
done
