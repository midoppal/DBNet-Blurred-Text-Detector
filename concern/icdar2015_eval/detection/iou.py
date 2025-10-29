#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon

## --------------------------- Pairwise / one-to-one eval --------------------------- ##
import os
import numpy as np


# EVAL_MATCH_MODE = os.getenv("EVAL_MATCH_MODE", "one_to_one").lower()  # "one_to_one" | "pairwise"
# EVAL_IOU_THR    = float(os.getenv("EVAL_IOU_THR", "0.5"))
# IGNORE_IOF_THR  = float(os.getenv("IGNORE_IOF_THR", "0.5"))  # det ignored if IOF w/ any dontcare >= this


DEFAULT_MATCH_MODE = "pairwise"   # "pairwise" | "one_to_one"
DEFAULT_IOU_THR = 0.5
DEFAULT_IGNORE_IOF_THR = 0.5   # a det is ignored if IOF(det | ###) >= this

def _polygon_area(poly):
    x = np.asarray(poly, dtype=np.float64)
    s = 0.0

    for i in range(len(x)):
        j = (i + 1) % len(x)
        s += x[i,0] * x[j,1] - x[j,0] * x[i,1]

    return abs(s) * 0.5

def _ioa(det_poly, gt_poly, denom="det", iou_func=None):

    """Intersection-over-area for ignore filtering.
    denom: "det" (IOF) or "gt".
    """

    A = _polygon_area(det_poly)
    B = _polygon_area(gt_poly)

    if A <= 0 or B <= 0:
        return 0.0
    
    if iou_func is None:
        from .iou import iou as iou_func  # your existing IoU
    iou_val = float(iou_func(det_poly, gt_poly))

    if iou_val <= 0.0:
        inter = 0.0

    else:
        # IoU = inter / (A + B - inter)  ->  inter = IoU * (A+B) / (1+IoU)
        inter = iou_val * (A + B) / (1.0 + iou_val)
    denom_area = A if denom == "det" else B

    return inter / denom_area

def _iou_matrix(gt_polys, det_polys, iou_func):
    n, m = len(gt_polys), len(det_polys)
    M = np.zeros((n, m), dtype=np.float32)
    for i, g in enumerate(gt_polys):
        for j, d in enumerate(det_polys):
            M[i, j] = float(iou_func(g, d))
    return M

def _mark_ignored_dets(det_polys, dontcare_polys, ignore_iof_thr, iou_func):
    if not dontcare_polys:
        return [False] * len(det_polys)
    ignore = []
    for d in det_polys:
        ign = any(_ioa(d, dc, denom="det", iou_func=iou_func) >= ignore_iof_thr
                  for dc in dontcare_polys)
        ignore.append(ign)
    return ignore

def evaluate_pairs(gt_polys, det_polys, dontcare_polys=None,
                   iou_thr=DEFAULT_IOU_THR,
                   match_mode=DEFAULT_MATCH_MODE,
                   ignore_iof_thr=DEFAULT_IGNORE_IOF_THR,
                   iou_func=None):
    """
    Returns dict: {tp, fp, fn, pairs, ignored_det_mask}
    pairs: list of (gt_idx, det_idx) credited as TP.
    """
    dontcare_polys = dontcare_polys or []
    if iou_func is None:
        from .iou import iou as iou_func

    # filter dets that mostly hit '###'
    det_ignored = _mark_ignored_dets(det_polys, dontcare_polys, ignore_iof_thr, iou_func)
    valid_det_idx = [j for j in range(len(det_polys)) if not det_ignored[j]]
    M = _iou_matrix(gt_polys, [det_polys[j] for j in valid_det_idx], iou_func)

    n, m = len(gt_polys), len(valid_det_idx)

    if match_mode == "pairwise":
        pairs = []
        gt_has = np.zeros(n, dtype=bool)
        det_has = np.zeros(m, dtype=bool)
        for i in range(n):
            for jj in range(m):
                if M[i, jj] >= iou_thr:
                    pairs.append((i, valid_det_idx[jj]))
                    gt_has[i] = True
                    det_has[jj] = True
        tp = len(pairs)
        fn = int((~gt_has).sum())
        fp = int((~det_has).sum())
        return dict(tp=tp, fp=fp, fn=fn, pairs=pairs, ignored_det_mask=det_ignored)

    # ---- one_to_one (ICDAR-like greedy by IoU) ----
    triples = [(M[i, jj], i, jj)
               for i in range(n) for jj in range(m) if M[i, jj] >= iou_thr]
    triples.sort(reverse=True, key=lambda x: x[0])

    used_gt = np.zeros(n, dtype=bool)
    used_det = np.zeros(m, dtype=bool)
    pairs = []
    for val, i, jj in triples:
        if not used_gt[i] and not used_det[jj]:
            used_gt[i] = True
            used_det[jj] = True
            pairs.append((i, valid_det_idx[jj]))

    tp = len(pairs)
    fn = int((~used_gt).sum())
    fp = int((~used_det).sum())
    return dict(tp=tp, fp=fp, fn=fn, pairs=pairs, ignored_det_mask=det_ignored)

## --------------------------- Pairwise / one-to-one eval --------------------------- ##

class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.3, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
        self.match_mode = "pairwise"#"pairwise"   # or "one_to_one"
        self.allow_union_if_already_matched = True

    def evaluate_image(self, gt, pred):

        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct)/(n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP
        match_mode = getattr(self, "match_mode", "pairwise")  # "pairwise" or "one_to_one"
        
        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols)-1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(
            gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols)-1)
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(
            detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")
        tp_gt = 0
        tp_det = 0
        tp_pair = 0
        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
            # --- REPLACEMENT: support "pairwise" and "one_to_one" ---
            gtCareSet  = set(range(len(gtPols))) - set(gtDontCarePolsNum)
            detCareSet = set(range(len(detPols))) - set(detDontCarePolsNum)

            # initialize pairwise counters so they're always defined
            tp_gt = tp_det = tp_pair = 0

            if match_mode == "pairwise":
                # Many-to-many credit: every IoU >= thresh is a TP pair
                gt_has = np.zeros(len(gtPols), dtype=np.uint8)
                det_has = np.zeros(len(detPols), dtype=np.uint8)

                for gtNum in gtCareSet:
                    for detNum in detCareSet:
                        if iouMat[gtNum, detNum] >= self.iou_constraint:
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            gt_has[gtNum] = 1
                            det_has[detNum] = 1
                            evaluationLog += f"Pairwise TP: GT #{gtNum} ↔ Det #{detNum}\n"

                # ========= UNION-OF-GTS PATCH (add this block) =========
                per_gt_cover_thr = getattr(self, "per_gt_cover_thr", 0.5)  # tunable knob

                # Track which (gt, det) pairs we already credited so we don't duplicate
                credited = set((p['gt'], p['det']) for p in pairs)

                # For each det that didn't individually match any GT at >= thr,
                # try to credit it against the UNION of overlapping GTs.
                for detNum in detCareSet:
                    if det_has[detNum] and not getattr(self, "allow_union_if_already_matched", False):
                        continue  # this det already matched at least one GT

                    det_poly = Polygon(detPols[detNum])

                    # Collect GTs that overlap this det and are sufficiently covered
                    overlapping_gts = []
                    for gtNum in gtCareSet:
                        gt_poly = Polygon(gtPols[gtNum])
                        inter = det_poly.intersection(gt_poly).area
                        if inter <= 0.0:
                            continue
                        gt_area = max(gt_poly.area, 1e-12)
                        gt_cover = inter / gt_area
                        if gt_cover >= per_gt_cover_thr:
                            overlapping_gts.append(gtNum)

                    if not overlapping_gts:
                        continue

                    # IoU between det and the UNION of those GTs
                    union_gt_poly = Polygon()
                    for gtNum in overlapping_gts:
                        union_gt_poly = union_gt_poly.union(Polygon(gtPols[gtNum]))

                    inter_u = det_poly.intersection(union_gt_poly).area
                    union_u = det_poly.union(union_gt_poly).area
                    iou_union = inter_u / max(union_u, 1e-12)

                    if iou_union >= self.iou_constraint:
                        # Credit this det to all overlapping GTs (that passed per-GT coverage)
                        for gtNum in overlapping_gts:
                            if (gtNum, detNum) not in credited:
                                pairs.append({'gt': gtNum, 'det': detNum})
                                credited.add((gtNum, detNum))
                                gt_has[gtNum] = 1
                                det_has[detNum] = 1
                                evaluationLog += (
                                    f"Union TP: Det #{detNum} ↔ GT #{gtNum} "
                                    f"(IoU_union={iou_union:.3f})\n"
                                )
                # ========= END UNION-OF-GTS PATCH =========

                # ========= UNION-OF-DETS PATCH (GT matched by the union of several dets) =========
                per_det_inside_gt_thr = getattr(self, "per_det_inside_gt_thr", 0.5)   # min fraction of each det inside this GT
                credit_all_dets_in_union = getattr(self, "credit_all_dets_in_union", True)  # credit all or just the best one

                # Rebuild credited set to avoid duplicate pairs
                credited = set((p['gt'], p['det']) for p in pairs)

                for gtNum in gtCareSet:
                    if gt_has[gtNum]:
                        continue  # already matched (individually or via union-of-GTs)
                    gt_poly = Polygon(gtPols[gtNum])

                    # Collect dets that overlap GT and are sufficiently "inside" it
                    candidate_dets = []
                    for detNum in detCareSet:
                        det_poly = Polygon(detPols[detNum])
                        inter = det_poly.intersection(gt_poly).area
                        if inter <= 0.0:
                            continue
                        det_area = max(det_poly.area, 1e-12)
                        det_inside = inter / det_area
                        if det_inside >= per_det_inside_gt_thr:
                            candidate_dets.append(detNum)

                    if not candidate_dets:
                        continue

                    # Union of candidate dets vs this GT
                    union_det_poly = Polygon()
                    for detNum in candidate_dets:
                        union_det_poly = union_det_poly.union(Polygon(detPols[detNum]))

                    inter_u = union_det_poly.intersection(gt_poly).area
                    union_u = union_det_poly.union(gt_poly).area
                    iou_union = inter_u / max(union_u, 1e-12)

                    if iou_union >= self.iou_constraint:
                        if credit_all_dets_in_union:
                            used_set = candidate_dets
                        else:
                            # credit only the single best det (largest intersection with this GT)
                            best = max(candidate_dets, key=lambda k: Polygon(detPols[k]).intersection(gt_poly).area)
                            used_set = [best]
                        for detNum in used_set:
                            if (gtNum, detNum) not in credited:
                                pairs.append({'gt': gtNum, 'det': detNum})
                                credited.add((gtNum, detNum))
                                gt_has[gtNum] = 1
                                det_has[detNum] = 1
                                evaluationLog += (
                                    f"UnionDet TP: GT #{gtNum} ↔ Det #{detNum} "
                                    f"(IoU_union={iou_union:.3f})\n"
                                )
                # ========= END UNION-OF-DETS PATCH =========

                # Now compute counts (must come AFTER the patch so added pairs are included)
                tp_gt   = int(gt_has.sum())   # unique GT matched
                tp_det  = int(det_has.sum())  # unique DET matched
                tp_pair = len(pairs)          # total credited pairs
                detMatched = tp_gt            # legacy compatibility

            else:
                # ********** LEGACY ONE-TO-ONE: EXACTLY as before **********
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if (gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and
                            gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum):
                            if iouMat[gtNum, detNum] > self.iou_constraint:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detMatched += 1
                                pairs.append({'gt': gtNum, 'det': detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                # keep tp_* aligned to legacy so dict always has keys
                tp_gt = tp_det = tp_pair = detMatched
            # --- REPLACEMENT: support "pairwise" and "one_to_one" ---

            # for gtNum in range(len(gtPols)):
            #     for detNum in range(len(detPols)):
            #         if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
            #             if iouMat[gtNum, detNum] > self.iou_constraint:
            #                 gtRectMat[gtNum] = 1
            #                 detRectMat[detNum] = 1
            #                 detMatched += 1
            #                 pairs.append({'gt': gtNum, 'det': detNum})
            #                 detMatchedNums.append(detNum)
            #                 evaluationLog += "Match GT #" + \
            #                     str(gtNum) + " with Det #" + str(detNum) + "\n"
        #---------------------------- Replacement ----------------------------#
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))

        if numGtCare == 0:
            recall = 1.0
            precision = 0.0 if numDetCare > 0 else 1.0
        else:
            if match_mode == "pairwise":
                # recall by unique GT matched; precision by unique DET matched
                recall = float(tp_gt) / numGtCare
                precision = 0.0 if numDetCare == 0 else float(tp_det) / numDetCare
            else:
                # LEGACY: both use detMatched
                recall = float(detMatched) / numGtCare
                precision = 0.0 if numDetCare == 0 else float(detMatched) / numDetCare
        #---------------------------- Replacement ----------------------------#

        # numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        # numDetCare = (len(detPols) - len(detDontCarePolsNum))
        # if numGtCare == 0:
        #     recall = float(1)
        #     precision = float(0) if numDetCare > 0 else float(1)
        # else:
        #     recall = float(detMatched) / numGtCare
        #     precision = 0 if numDetCare == 0 else float(
        #         detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
            precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'detMatched': detMatched,
            'evaluationLog': evaluationLog,
            'match_mode': match_mode,          # <-- ADD THIS LINE
            'tp_gt': tp_gt,       # = detMatched in one_to_one
            'tp_det': tp_det,     # = detMatched in one_to_one
            'tp_pairs': tp_pair,  # = detMatched in one_to_one
        }

        return perSampleMetrics

    # def combine_results(self, results):
    #     numGlobalCareGt = 0
    #     numGlobalCareDet = 0
    #     matchedSum = 0
    #     for result in results:
    #         numGlobalCareGt += result['gtCare']
    #         numGlobalCareDet += result['detCare']
    #         matchedSum += result['detMatched']

    #     methodRecall = 0 if numGlobalCareGt == 0 else float(
    #         matchedSum)/numGlobalCareGt
    #     methodPrecision = 0 if numGlobalCareDet == 0 else float(
    #         matchedSum)/numGlobalCareDet
    #     methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
    #         methodRecall * methodPrecision / (methodRecall + methodPrecision)

    #     methodMetrics = {'precision': methodPrecision,
    #                      'recall': methodRecall, 'hmean': methodHmean}

    #     return methodMetrics
    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        all_o2o = True
        for r in results:
            if r.get('match_mode', 'one_to_one') != 'one_to_one':
                all_o2o = False
                break

        if all_o2o:
            # ***** LEGACY aggregation *****
            matchedSum = 0
            for r in results:
                numGlobalCareGt  += r['gtCare']
                numGlobalCareDet += r['detCare']
                matchedSum       += r['detMatched']
            recall    = 0 if numGlobalCareGt  == 0 else float(matchedSum)/numGlobalCareGt
            precision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
            hmean = 0 if recall + precision == 0 else 2*recall*precision/(recall+precision)
            return {'precision': precision, 'recall': recall, 'hmean': hmean}

        # ***** Pairwise-safe aggregation *****
        matchedSumGt = 0
        matchedSumDet = 0
        for r in results:
            numGlobalCareGt  += r['gtCare']
            numGlobalCareDet += r['detCare']
            matchedSumGt     += int(r.get('tp_gt',  r['detMatched']))
            matchedSumDet    += int(r.get('tp_det', r['detMatched']))
        recall    = 0 if numGlobalCareGt  == 0 else float(matchedSumGt)  / numGlobalCareGt
        precision = 0 if numGlobalCareDet == 0 else float(matchedSumDet) / numGlobalCareDet
        hmean = 0 if recall + precision == 0 else 2*recall*precision/(recall+precision)
        return {'precision': precision, 'recall': recall, 'hmean': hmean}

if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
