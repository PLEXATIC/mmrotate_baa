_base_ = [
    '../projects/RR360/configs360/_base_/datasets/dota.py', '../projects/RR360/configs360/_base_/schedules/schedule_1x.py',
    '../projects/RR360/configs360/_base_/default_runtime.py'
]
angle_version = 'r360'

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

METAINFO = dict(classes=tuple(classes))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,
        save_best=['dota/AP50'],
        rule='greater',
        max_keep_ckpts=1), )

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.0,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'DOTADataset'

model = dict(
    type='mmdet.RetinaNet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[125.0, 125.0, 125.0],
        std=[125.0, 125.0, 125.0],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='mmdet.RetinaHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(
            type='mmdet.PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_root="./",
        ann_file='datasets/dota_all_train_val_ir_correct_names/labelTxt/',
        data_prefix=dict(img_path='datasets/dota_all_train_val_ir_correct_names/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_root="./",
        ann_file='datasets/dota_all_train_val_ir_correct_names/validation_labelTxt/',
        data_prefix=dict(img_path='datasets/dota_all_train_val_ir_correct_names/validation_images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='DOTAR360Metric', metric='mAP', iou_thrs=[0.5], angle_thr=90),
    dict(type='DOTAR360Metric', metric='mAP', iou_thrs=[0.5], angle_thr=360),
    dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.5])
]
test_evaluator = [
    dict(
        type='DOTAR360Metric',
        metric='mAP',
        iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        angle_thr=90),
    dict(
        type='DOTAMetric',
        metric='mAP',
        iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
]