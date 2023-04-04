dataset_type = 'DOTADataset'
img_norm_cfg = dict(
    mean=[125.0, 125.0, 125.0], std=[125.0, 125.0, 125.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(640, 640)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.75,
        angles_range=180,
        allow_negative=True),
    dict(
        type='Normalize',
        mean=[125.0, 125.0, 125.0],
        std=[125.0, 125.0, 125.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[125.0, 125.0, 125.0],
                std=[125.0, 125.0, 125.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='DOTADataset',
        ann_file='./datasets/dota_all_train_val_ir_correct_names/labelTxt/',
        img_prefix='./datasets/dota_all_train_val_ir_correct_names/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(640, 640)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='le90'),
            dict(
                type='PolyRandomRotate',
                rotate_ratio=0.75,
                angles_range=180,
                allow_negative=True),
            dict(
                type='Normalize',
                mean=[125.0, 125.0, 125.0],
                std=[125.0, 125.0, 125.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        version='le90',
        classes=[
            'black_shatter', 'cylinder_globe', 'black_chair', 'misc_star',
            'cylinder_shopping', 'black_a', 'blue_location', 'black_family',
            'misc_staircase', 'blue_house_small', 'blue_handplant',
            'misc_plant_spikey', 'black_scale', 'black_movie', 'blue_palace',
            'misc_plant_round', 'blue_temple', 'black_tasklist',
            'blue_shopping_bags', 'cylinder_coconut_tree', 'blue_church',
            'blue_x_and_o', 'misc_v_wood_gray', 'misc_coffee_cup',
            'misc_concrete_v', 'black_math_operators', 'black_stacked_dials',
            'blue_coconut_tree', 'black_ti', 'cylinder_shopping_cart',
            'misc_gray_vv', 'cylinder_flyer_card', 'blue_house_big_one_sided',
            'blue_scyscraper', 'black_v'
        ]),
    val=dict(
        type='DOTADataset',
        ann_file=
        './datasets/dota_all_train_val_ir_correct_names/validation_labelTxt/',
        img_prefix=
        './datasets/dota_all_train_val_ir_correct_names/validation_images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[125.0, 125.0, 125.0],
                        std=[125.0, 125.0, 125.0],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90',
        classes=[
            'black_shatter', 'cylinder_globe', 'black_chair', 'misc_star',
            'cylinder_shopping', 'black_a', 'blue_location', 'black_family',
            'misc_staircase', 'blue_house_small', 'blue_handplant',
            'misc_plant_spikey', 'black_scale', 'black_movie', 'blue_palace',
            'misc_plant_round', 'blue_temple', 'black_tasklist',
            'blue_shopping_bags', 'cylinder_coconut_tree', 'blue_church',
            'blue_x_and_o', 'misc_v_wood_gray', 'misc_coffee_cup',
            'misc_concrete_v', 'black_math_operators', 'black_stacked_dials',
            'blue_coconut_tree', 'black_ti', 'cylinder_shopping_cart',
            'misc_gray_vv', 'cylinder_flyer_card', 'blue_house_big_one_sided',
            'blue_scyscraper', 'black_v'
        ]),
    test=dict(
        type='DOTADataset',
        ann_file=
        './datasets/dota_all_train_val_ir_correct_names/test_labelTxt/',
        img_prefix=
        './datasets/dota_all_train_val_ir_correct_names/test_images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[125.0, 125.0, 125.0],
                        std=[125.0, 125.0, 125.0],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90',
        classes=[
            'black_shatter', 'cylinder_globe', 'black_chair', 'misc_star',
            'cylinder_shopping', 'black_a', 'blue_location', 'black_family',
            'misc_staircase', 'blue_house_small', 'blue_handplant',
            'misc_plant_spikey', 'black_scale', 'black_movie', 'blue_palace',
            'misc_plant_round', 'blue_temple', 'black_tasklist',
            'blue_shopping_bags', 'cylinder_coconut_tree', 'blue_church',
            'blue_x_and_o', 'misc_v_wood_gray', 'misc_coffee_cup',
            'misc_concrete_v', 'black_math_operators', 'black_stacked_dials',
            'blue_coconut_tree', 'black_ti', 'cylinder_shopping_cart',
            'misc_gray_vv', 'cylinder_flyer_card', 'blue_house_big_one_sided',
            'blue_scyscraper', 'black_v'
        ]))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='Adam', lr=0.003, weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
angle_version = 'le90'
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=35,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
classes = [
    'black_shatter', 'cylinder_globe', 'black_chair', 'misc_star',
    'cylinder_shopping', 'black_a', 'blue_location', 'black_family',
    'misc_staircase', 'blue_house_small', 'blue_handplant',
    'misc_plant_spikey', 'black_scale', 'black_movie', 'blue_palace',
    'misc_plant_round', 'blue_temple', 'black_tasklist', 'blue_shopping_bags',
    'cylinder_coconut_tree', 'blue_church', 'blue_x_and_o', 'misc_v_wood_gray',
    'misc_coffee_cup', 'misc_concrete_v', 'black_math_operators',
    'black_stacked_dials', 'blue_coconut_tree', 'black_ti',
    'cylinder_shopping_cart', 'misc_gray_vv', 'cylinder_flyer_card',
    'blue_house_big_one_sided', 'blue_scyscraper', 'black_v'
]
work_dir = './work_dirs/custom_config2_ir_rotaug_adam'
auto_resume = False
gpu_ids = range(0, 1)
