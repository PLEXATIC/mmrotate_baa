
# the new config inherits the base configs to highlight the necessary modification
_base_ = './rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py'

# 1. dataset settings
dataset_type = 'DOTADataset'
classes = ('toke-nr-1',
        'toke-nr-2',
        'toke-nr-3',
        'toke-nr-4',
        'toke-nr-5',
        'toke-nr-6',
        'toke-nr-7',
        'toke-nr-8',
        'toke-nr-9',
        'toke-nr-10',
        'toke-nr-11',
        'toke-nr-12',
        'toke-nr-13',
        'toke-nr-14',
        'toke-nr-15',
        'toke-nr-16',
        'toke-nr-17',
        'toke-nr-18',
        'toke-nr-19',
        'toke-nr-20',
        'toke-nr-21',
        'toke-nr-22',
        'toke-nr-23',
        'toke-nr-24',
        'toke-nr-25',
        'toke-nr-26',
        'toke-nr-27',
        'toke-nr-28',
        'toke-nr-29',
        'toke-nr-30',
        'toke-nr-31',
        'toke-nr-32',
        'toke-nr-33',
        'toke-nr-34',
        'toke-nr-0')

img_norm_cfg = dict(
    mean=[125.0, 125.0, 125.0], std=[125.0, 125.0, 125.0], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./datasets/dota_all/labelTxt/',
        img_prefix='./datasets/dota_all/images/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./datasets/dota_all/labelTxt/',
        img_prefix='./datasets/dota_all/images/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='./datasets/dota_all/labelTxt/',
        img_prefix='./datasets/dota_all/images/'))

model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=len(classes)))