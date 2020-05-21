# this script runs 5 replica experiments for non adversarial training and for adversarially robust training across all 4 different attacks implemented in this framework

# non adversarial training of base classifier
python train_classifier.py --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_1_mobilenet_v2_no_adv --tb_summaries_dir tb_1_mobilenet_v2_no_adv --gpu_id 1
python train_classifier.py --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_2_mobilenet_v2_no_adv --tb_summaries_dir tb_2_mobilenet_v2_no_adv --gpu_id 1
python train_classifier.py --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_3_mobilenet_v2_no_adv --tb_summaries_dir tb_3_mobilenet_v2_no_adv --gpu_id 1
python train_classifier.py --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_4_mobilenet_v2_no_adv --tb_summaries_dir tb_4_mobilenet_v2_no_adv --gpu_id 1
python train_classifier.py --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_5_mobilenet_v2_no_adv --tb_summaries_dir tb_5_mobilenet_v2_no_adv --gpu_id 1

# adversarially robust classifier
python train_robust.py --attack_type linfpgd --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_1_mobilenet_v2_adv_linfpgd --tb_summaries_dir tb_1_mobilenet_v2_adv_linf_pgd --gpu_id 1
python train_robust.py --attack_type linfpgd --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_2_mobilenet_v2_adv_linfpgd --tb_summaries_dir tb_2_mobilenet_v2_adv_linf_pgd --gpu_id 1
python train_robust.py --attack_type linfpgd --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_3_mobilenet_v2_adv_linfpgd --tb_summaries_dir tb_3_mobilenet_v2_adv_linf_pgd --gpu_id 1
python train_robust.py --attack_type linfpgd --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_4_mobilenet_v2_adv_linfpgd --tb_summaries_dir tb_4_mobilenet_v2_adv_linf_pgd --gpu_id 1
python train_robust.py --attack_type linfpgd --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_5_mobilenet_v2_adv_linfpgd --tb_summaries_dir tb_5_mobilenet_v2_adv_linf_pgd --gpu_id 1
#python train_robust.py --attack_type gsa --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_1_mobilenet_v2_adv_gsa --tb_summaries_dir tb_1_mobilenet_v2_adv_gsa --gpu_id 1
python train_robust.py --attack_type gsa --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_2_mobilenet_v2_adv_gsa --tb_summaries_dir tb_2_mobilenet_v2_adv_gsa --gpu_id 1
python train_robust.py --attack_type gsa --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_3_mobilenet_v2_adv_gsa --tb_summaries_dir tb_3_mobilenet_v2_adv_gsa --gpu_id 1
python train_robust.py --attack_type gsa --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_4_mobilenet_v2_adv_gsa --tb_summaries_dir tb_4_mobilenet_v2_adv_gsa --gpu_id 1
python train_robust.py --attack_type gsa --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_5_mobilenet_v2_adv_gsa --tb_summaries_dir tb_5_mobilenet_v2_adv_gsa --gpu_id 1
python train_robust.py --attack_type singlepixel --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_1_mobilenet_v2_adv_singlepixel --tb_summaries_dir tb_1_mobilenet_v2_adv_singlepixel --gpu_id 1
python train_robust.py --attack_type singlepixel --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_2_mobilenet_v2_adv_singlepixel --tb_summaries_dir tb_2_mobilenet_v2_adv_singlepixel --gpu_id 1
python train_robust.py --attack_type singlepixel --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_3_mobilenet_v2_adv_singlepixel --tb_summaries_dir tb_3_mobilenet_v2_adv_singlepixel --gpu_id 1
python train_robust.py --attack_type singlepixel --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_4_mobilenet_v2_adv_singlepixel --tb_summaries_dir tb_4_mobilenet_v2_adv_singlepixel --gpu_id 1
python train_robust.py --attack_type singlepixel --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_5_mobilenet_v2_adv_singlepixel --tb_summaries_dir tb_5_mobilenet_v2_adv_singlepixel --gpu_id 1
python train_robust.py --attack_type jacobiansaliencymap --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_1_mobilenet_v2_adv_jacobiansaliencymap --tb_summaries_dir tb_1_mobilenet_v2_adv_jacobiansaliencymap --gpu_id 1
python train_robust.py --attack_type jacobiansaliencymap --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_2_mobilenet_v2_adv_jacobiansaliencymap --tb_summaries_dir tb_2_mobilenet_v2_adv_jacobiansaliencymap --gpu_id 1
python train_robust.py --attack_type jacobiansaliencymap --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_3_mobilenet_v2_adv_jacobiansaliencymap --tb_summaries_dir tb_3_mobilenet_v2_adv_jacobiansaliencymap --gpu_id 1
python train_robust.py --attack_type jacobiansaliencymap --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_4_mobilenet_v2_adv_jacobiansaliencymap --tb_summaries_dir tb_4_mobilenet_v2_adv_jacobiansaliencymap --gpu_id 1
python train_robust.py --attack_type jacobiansaliencymap --dataset_root 'ddos_dataset' --chkpt_dir checkpoint_5_mobilenet_v2_adv_jacobiansaliencymap --tb_summaries_dir tb_5_mobilenet_v2_adv_jacobiansaliencymap --gpu_id 1

