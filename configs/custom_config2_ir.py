
# the new config inherits the base configs to highlight the necessary modification
_base_ = './rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py'

# 1. dataset settings
dataset_type = 'DOTADataset'
classes = [
    'black_shatter',
    'cylinder_globe',
    'black_chair',
    'misc_star',
    'cylinder_shopping',
    'black_a',
    'blue_location',
    'black_family',
    'misc_staircase',
    'blue_house_small',
    'blue_handplant',
    'misc_plant_spikey',
    'black_scale',
    'black_movie',
    'blue_palace',
    'misc_plant_round',
    'blue_temple',
    'black_tasklist',
    'blue_shopping_bags',
    'cylinder_coconut_tree',
    'blue_church',
    'blue_x_and_o',
    'misc_v_wood_gray',
    'misc_coffee_cup',
    'misc_concrete_v',
    'black_math_operators',
    'black_stacked_dials',
    'blue_coconut_tree',
    'black_ti',
    'cylinder_shopping_cart',
    'misc_gray_vv',
    'cylinder_flyer_card',
    'blue_house_big_one_sided',
    'blue_scyscraper',
    'black_v'
]

img_norm_cfg = dict(
    mean=[125.0, 125.0, 125.0], std=[125.0, 125.0, 125.0], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./datasets/dota_all_train_val_ir_correct_names/labelTxt/',
        img_prefix='./datasets/dota_all_train_val_ir_correct_names/images/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./datasets/dota_all_train_val_ir_correct_names/validation_labelTxt/',
        img_prefix='./datasets/dota_all_train_val_ir_correct_names/validation_images/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./datasets/dota_all_train_val_ir_correct_names/test_labelTxt/',
        img_prefix='./datasets/dota_all_train_val_ir_correct_names/test_images/'))

model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=len(classes)))