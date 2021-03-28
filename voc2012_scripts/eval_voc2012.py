import _add_project_path

import os
import tqdm
import pickle
import tensorflow as tf
from termcolor import colored
from absl import flags, app
from pascalvoc_ap.ap import get_ap
from libs.utils import yolo_output2boxes, box_postp2use
from datasets.voc2012_tfds.voc2012 import GetVoc2012
from datasets.voc2012_tfds.libs import prep_voc_data, VOC_CLS_MAP
from configs import ProjectPath, cfg


FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', default=cfg.batch_size, help='Batch size')
flags.DEFINE_string('pb_dir', default=os.path.join(ProjectPath.VOC2012_CKPTS_DIR.value, 'yolo_voc2012_448x448'), help='Save pb directory path')


def main(argv):
    yolo = tf.saved_model.load(
        export_dir=FLAGS.pb_dir,
        tags=None,
        options=None,
    )

    voc2012 = GetVoc2012(batch_size=FLAGS.batch_size)
    voc2012_val_gts_all_path = os.path.join(ProjectPath.DATASETS_DIR.value, 'voc2012_tfds', 'eval', 'val_gts_all_448_full.pickle')
    if os.path.exists(voc2012_val_gts_all_path):
        voc2012_val_gts_all = pickle.load(open(voc2012_val_gts_all_path, 'rb'))
        cls_name_list = list(VOC_CLS_MAP.values())

    val_ds = voc2012.get_val_ds()
    img_id = 0
    val_preds_all = list()

    for step, batch_data in tqdm.tqdm(enumerate(val_ds, 1), total=len(val_ds), desc='Validation'):
        batch_imgs, batch_labels = prep_voc_data(batch_data, input_height=cfg.input_height, input_width=cfg.input_width)
        yolo_output_raw = yolo(batch_imgs)
        yolo_boxes = yolo_output2boxes(yolo_output_raw, cfg.input_height, cfg.input_width, cfg.cell_size, cfg.boxes_per_cell)
        for i in range(len(yolo_boxes)):
            output_boxes = box_postp2use(yolo_boxes[i], cfg.nms_iou_thr, 0.)
            if output_boxes.size == 0:
                img_id += 1
                continue
            for output_box in output_boxes:
                *pts, conf, cls_idx = output_box
                cls_name = VOC_CLS_MAP[cls_idx]
                val_preds_all.append([cls_name, conf, *map(round, pts), img_id])
            img_id += 1

    APs = get_ap(preds_all=val_preds_all, gts_all=voc2012_val_gts_all, classes=cls_name_list, iou_thr=0.5)
    mAP = APs.pop('mAP')
    APs_log = '\n====== mAP ======\n' + f'* mAP: {mAP:<8.4f}\n'
    for cls_name, ap in APs.items():
        APs_log += f'- {cls_name}: {ap:<8.4f}\n'
    APs_log += '====== ====== ======\n'
    APs_log_colored = colored(APs_log, 'magenta')
    print(APs_log_colored)
    

if __name__ == '__main__':
    app.run(main)
