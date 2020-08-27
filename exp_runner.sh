dataset_path=${1}
checkpoint_dir=${2}
tb_summaries_dir=${3}
gpu_id=${4:-1}
replica_num=${5:-5}

# make necessary dirs
mkdir ${checkpoint_dir}
mkdir ${tb_summaries_dir}

declare -a model_type=('mobilenet_v2' 'resnet34')

declare -a attack_type=('linfpgd'
                        'gsa'
                        'singlepixel'
                        'all'
                       )

# train classifier with no adversary
for model in "${model_type[@]}"
do
    for idx in $(seq 1 $replica_num)
    do
        echo $model
        echo $idx
        python train_classifier.py --model_type $model --dataset_root $dataset_path --chkpt_dir "${checkpoint_dir}/${idx}_${model}_no_adv" --tb_summaries_dir "${tb_summaries_dir}/${idx}_${model}_no_adv" --gpu_id 1
    done
done

# train adversarially robust classifier with different adversaries
for model in "${model_type[@]}"
do
    for attack in "${attack_type[@]}"
    do
        for idx in $(seq 1 $replica_num)
        do
            python train_robust.py --model_type $model --attack_type $attack --dataset_root $dataset_path --chkpt_dir "${checkpoint_dir}/${idx}_${model}_${attack}" --tb_summaries_dir "${tb_summaries_dir}/${idx}_${model}_${attack}" --gpu_id 1
        done
    done
done
