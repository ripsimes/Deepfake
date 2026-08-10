from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # Fix 1: ensure compatible 1D arrays of same length
        image = np.array(image).flatten()
        pred = np.array(pred).flatten()
        label = np.array(label).flatten()
        n = min(len(image), len(pred), len(label))
        image, pred, label = image[:n], pred[:n], label[:n]
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            # Fix 2: use filename as key so flat image datasets
            # (all images in one folder) don't collapse to a single "video"
            a = parts[-1] if len(parts) < 2 else parts[-2]
            if a not in result_dict:
                result_dict[a] = []
            result_dict[a].append(item)
        image_arr = list(result_dict.values())
        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        # Fix 3: guard against degenerate ROC (single class or single sample)
        if len(set(new_label)) < 2:
            return float('nan'), float('nan')
        try:
            fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
            v_auc = metrics.auc(fpr, tpr)
            fnr = 1 - tpr
            v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        except (ValueError, IndexError):
            v_auc = float('nan')
            v_eer = float('nan')
        return v_auc, v_eer

    y_pred = y_pred.squeeze()
    y_true[y_true >= 1] = 1
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    ap = metrics.average_precision_score(y_true, y_pred)
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    f1 = metrics.f1_score(np.clip(y_true, 0, 1), prediction_class, zero_division=0)
    if type(img_names[0]) is not list:
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
        # Fix 4: fall back to frame-level AUC if video grouping fails
        if np.isnan(v_auc):
            v_auc = auc
    else:
        v_auc = auc
    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'f1': f1, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
