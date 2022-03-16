from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

annType = 'bbox'
cocoGt = COCO('/Users/cabe0006/Projects/monash/cvpr_experiments/nature_journal_publication/predictions_rcnn_test_5/coco_format/ground-truth-test.json')
cocoDt = cocoGt.loadRes('/Users/cabe0006/Projects/monash/cvpr_experiments/nature_journal_publication/predictions_rcnn_test_5/coco_format/test-predictions-test.json')
cocoEval = COCOeval(cocoGt, cocoDt, annType)
imgIds = sorted(cocoGt.getImgIds())
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

