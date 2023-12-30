CUDA='0,1,2,3,4,5,6,7'
N_GPU=8
BATCH=256
DATA=/data/ImageNetS/ImageNetS50
IMAGENETS=/data/ImageNetS/ImageNetS50

DUMP_PATH=./weights/pass50
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention
DUMP_PATH_FINETUNE_1=${DUMP_PATH}/pixel_attention_1
DUMP_PATH_FINETUNE_2=${DUMP_PATH}/pixel_attention_2
DUMP_PATH_classification=${DUMP_PATH}/pixel_classification
DUMP_PATH_classification1=${DUMP_PATH}/pixel_classification1
DUMP_PATH_classification2=${DUMP_PATH}/pixel_classification2
DUMP_PATH_csd=${DUMP_PATH}pixel_csd
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning

QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet18
NUM_CLASSES=50
EPOCH=400
EPOCH_PIXELATT=20
EPOCH_SEG=20
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_FINETUNE_1}
mkdir -p ${DUMP_PATH_FINETUNE_2}
mkdir -p ${DUMP_PATH_classification}
mkdir -p ${DUMP_PATH_classification1}
mkdir -p ${DUMP_PATH_classification2}

mkdir -p ${DUMP_PATH_SEG}

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pretrain.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH} \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH} \
--epoch_queue_starts 15 \
--epochs ${EPOCH} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 10 \
--seed 31 \
--shallow 3 \
--weights 1 1

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pixel_attention.py \
--arch ${ARCH} \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
--nmb_crops 2 \
--size_crops 224 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH_PIXELATT} \
--epoch_queue_starts 0 \
--epochs ${EPOCH_PIXELATT} \
--batch_size ${BATCH} \
--base_lr 6.0 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES_PIXELATT} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 10 \
--seed 31 \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar


#Three cluster centers are used in total
#cluster centers1
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
-c 50 \
--seed 31

#cluster centers2
#The pretraining weights for clustering are obtained based on the DFF
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification2}/ckp-.pth.tar \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE_1} \
-c 50 \
--seed 31

#cluster centers3
#The dataset train_csd is obtained by filtering and resampling trains based on CSD
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification2}/ckp-.pth.tar \
--data_path ${DUMP_PATH_csd}/train_csd \
--dump_path ${DUMP_PATH_FINETUNE_2} \
 -c 50 \
 --seed 31

#DFF
#the classification network f
CUDA_VISIBLE_DEVICES=${CUDA} python classification.py --arch ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar  \
--data_path ${IMAGENETS}/train \
--pseudo_label  ${DUMP_PATH_FINETUNE}/cluster/train_labeled.txt \
--dump_path ${DUMP_PATH_classification} \
--epochs 15 \
--batch_size 64 \
--base_lr 0.01 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes 50


#the classification network g
#The dataset train_S is a small loss sample selected by the network f and train_S.txt is the corresponding pseudo label
CUDA_VISIBLE_DEVICES=${CUDA} python classification.py --arch ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar  \
--data_path ${DUMP_PATH_classification}/sample_select/train_S \
--pseudo_label  ${DUMP_PATH_classification}/sample_select/train_S.txt \
--dump_path ${DUMP_PATH_classification1} \
--epochs 50 \
--batch_size 32 \
--base_lr 0.01 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes 50

#Infer training set labels
CUDA_VISIBLE_DEVICES=${CUDA} python inference_classification.py \
--a ${ARCH} \
--pretrained ${DUMP_PATH_classification}/checkpoints/ckp-36.pth.tar \
--data_path ${IMAGENETS}  \
--dump_path  ${DUMP_PATH_classification} \
-c 50 \
--mode train

#the classification network h
#The dataset train_S2 is a small loss sample selected by the network g and train_S2.txt is the corresponding pseudo label
CUDA_VISIBLE_DEVICES=${CUDA} python classification_clu.py --arch ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar  \
--data_path ${DUMP_PATH_classification1}/sample_select/train_S2 \
--pseudo_label  ${DUMP_PATH_classification1}/sample_select/train_S2.txt \
--dump_path ${DUMP_PATH_classification2} \
--epochs 50 \
--batch_size 32 \
--base_lr 0.01 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes 50




##### Evaluating the pseudo labels on the validation set.
CUDA_VISIBLE_DEVICES=0 python inference_pixel_attention.py -a resnet18 \
--pretrained ${DUMP_PATH_classification2}/ckp-.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE_2} \
-c ${NUM_CLASSES} \
--mode validation \
--test \
--centroid ${DUMP_PATH_FINETUNE_2}/cluster/centroids.npy


CUDA_VISIBLE_DEVICES=0 python evaluator.py \
--predict_path ${DUMP_PATH_FINETUNE_2} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation \
--curve \
--min 20 \
--max 80


CUDA_VISIBLE_DEVICES=${CUDA}  python ./inference_pixel_attention_1.py -a ${ARCH} \
--pretrained ${DUMP_PATH_classification2}/ckp-.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
--dump_path_1 ${DUMP_PATH_FINETUNE}/logit_max_train/ \
--testPoint_path ${DUMP_PATH_FINETUNE}/trainPoint.txt \
--testLabel_path ${DUMP_PATH_FINETUNE}/trainLabel.txt \
-c ${NUM_CLASSES} \
--mode train \
--centroid {DUMP_PATH_FINETUNE_2}/cluster/centroids.npy \
-t 0.41

python cat_txt.py

CUDA_VISIBLE_DEVICES=0 python ./SAM-SPO-jittor/spo.py \
--checkpoint ./SAM-jittor/checkpoint/sam_vit_b_01ec64.pth \
--model vit_b \
--input  ${IMAGENETS}/train \
--output ./SAM-SPO-jittor/spo/train  \
--output_mask ./SAM-SPO-jittor/spo/train-mask  \
--mask ${DUMP_PATH_FINETUNE}/train \
--txt1_path ${DUMP_PATH_FINETUNE}/trainPoint.txt \
--txt_path_class {DUMP_PATH_FINETUNE_2}/cluster/train_I_D_50.txt \
--outputbox ./SAM-SPO-jittor/spo/train-box \
--yz 0.8

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size 256 \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes ${NUM_CLASSES} \
--pseudo_path ./SAM-SPO-jittor/spo/train \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG}/validation/match.json

CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_SEG} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation