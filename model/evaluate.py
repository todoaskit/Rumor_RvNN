# -*- coding: utf-8 -*-
"""
@object: weibo & twitter
@task: split train & test, evaluate performance 
@author: majing
@variable: T, 
@time: Tue Nov 10 16:29:42 2015
"""


# evaluation of model result
def evaluation(prediction, y):
    # no. of time series
    path = '../eval'
    t_p = 0
    t_n = 0
    f_p = 0
    f_n = 0
    e = 0.000001
    threshold = 0.5
    f_out = open(path, 'w')
    for i in range(len(y)):
        f_out.write(str(y[i][0]) + "\t" + str(prediction[i][0]) + "\n")
        if y[i][0] == 1 and prediction[i][0] >= threshold:
            t_p += 1
        if y[i][0] == 1 and prediction[i][0] < threshold:
            f_n += 1
        if y[i][0] == 0 and prediction[i][0] >= threshold:
            f_p += 1
        if y[i][0] == 0 and prediction[i][0] < threshold:
            t_n += 1
    f_out.close()
    accuracy = float(t_p + t_n) / (t_p + t_n + f_p + f_n + e)
    precision_r = float(t_p) / (t_p + f_p + e)  # for rumor
    recall_r = float(t_p) / (t_p + f_n + e)
    f_measure_r = 2 * precision_r * recall_r / (precision_r + recall_r + e)
    precision_f = float(t_n) / (t_n + f_n + e)  # for fact
    recall_f = float(t_n) / (t_n + f_p + e)
    f_measure_f = 2 * precision_f * recall_f / (precision_f + recall_f + e)
    return [accuracy, precision_r, recall_r, f_measure_r, precision_f, recall_f, f_measure_f]


def evaluation_2class(prediction, y):  # 4 dim
    t_p1, f_p1, f_n1, t_n1 = 0, 0, 0, 0
    t_p2, f_p2, f_n2, t_n2 = 0, 0, 0, 0
    e, rmse, rmse_1, rmse_2 = 0.000001, 0.0, 0.0, 0.0
    for i in range(len(y)):
        y_i, p_i = list(y[i]), list(prediction[i][0])
        # rmse
        for j in range(len(y_i)):
            rmse += (y_i[j] - p_i[j]) ** 2
        rmse_1 += (y_i[0] - p_i[0]) ** 2
        rmse_2 += (y_i[1] - p_i[1]) ** 2
        # pre, Recall, F    
        act = str(y_i.index(max(y_i)) + 1)
        pre = str(p_i.index(max(p_i)) + 1)

        # for class 1
        if act == '1' and pre == '1':
            t_p1 += 1
        if act == '1' and pre != '1':
            f_n1 += 1
        if act != '1' and pre == '1':
            f_p1 += 1
        if act != '1' and pre != '1':
            t_n1 += 1

        # for class 2
        if act == '2' and pre == '2':
            t_p2 += 1
        if act == '2' and pre != '2':
            f_n2 += 1
        if act != '2' and pre == '2':
            f_p2 += 1
        if act != '2' and pre != '2':
            t_n2 += 1

    # print result
    accuracy_all = round(float(t_p1 + t_p2) / float(len(y) + e), 4)
    precision_1 = round(float(t_p1) / float(t_p1 + f_p1 + e), 4)
    recall_1 = round(float(t_p1) / float(t_p1 + f_n1 + e), 4)
    f1 = round(2 * precision_1 * recall_1 / (precision_1 + recall_1 + e), 4)

    precision_2 = round(float(t_p2) / float(t_p2 + f_p2 + e), 4)
    recall_2 = round(float(t_p2) / float(t_p2 + f_n2 + e), 4)
    f2 = round(2 * precision_2 * recall_2 / (precision_2 + recall_2 + e), 4)

    rmse_all = round((rmse / len(y)) ** 0.5, 4)
    rmse_all_1 = round((rmse_1 / len(y)) ** 0.5, 4)
    rmse_all_2 = round((rmse_2 / len(y)) ** 0.5, 4)

    rmse_all_avg = round((rmse_all_1 + rmse_all_2) / 2, 4)
    return [accuracy_all, rmse_all, rmse_all_avg, 'C1:', precision_1, precision_1, recall_1, f1, '\n',
            'C2:', precision_2, precision_2, recall_2, f2, '\n']


def evaluation_4class(prediction, y):  # 4 dim
    t_p1, f_p1, f_n1, t_n1 = 0, 0, 0, 0
    t_p2, f_p2, f_n2, t_n2 = 0, 0, 0, 0
    t_p3, f_p3, f_n3, t_n3 = 0, 0, 0, 0
    t_p4, f_p4, f_n4, t_n4 = 0, 0, 0, 0
    e, rmse, rmse1, rmse2, rmse3, rmse4 = 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(y)):
        y_i, p_i = list(y[i]), list(prediction[i][0])
        
        # rmse
        for j in range(len(y_i)):
            rmse += (y_i[j] - p_i[j]) ** 2
        rmse1 += (y_i[0] - p_i[0]) ** 2
        rmse2 += (y_i[1] - p_i[1]) ** 2
        rmse3 += (y_i[2] - p_i[2]) ** 2
        rmse4 += (y_i[3] - p_i[3]) ** 2
        
        # Pre, Recall, F    
        act = str(y_i.index(max(y_i)) + 1)
        pre = str(p_i.index(max(p_i)) + 1)

        # for class 1
        if act == '1' and pre == '1':
            t_p1 += 1
        if act == '1' and pre != '1':
            f_n1 += 1
        if act != '1' and pre == '1':
            f_p1 += 1
        if act != '1' and pre != '1':
            t_n1 += 1

        # for class 2
        if act == '2' and pre == '2':
            t_p2 += 1
        if act == '2' and pre != '2':
            f_n2 += 1
        if act != '2' and pre == '2':
            f_p2 += 1
        if act != '2' and pre != '2':
            t_n2 += 1

        # for class 3
        if act == '3' and pre == '3':
            t_p3 += 1
        if act == '3' and pre != '3':
            f_n3 += 1
        if act != '3' and pre == '3':
            f_p3 += 1
        if act != '3' and pre != '3':
            t_n3 += 1

        # for class 4
        if act == '4' and pre == '4':
            t_p4 += 1
        if act == '4' and pre != '4':
            f_n4 += 1
        if act != '4' and pre == '4':
            f_p4 += 1
        if act != '4' and pre != '4':
            t_n4 += 1
    
    # print result
    accuracy_all = round(float(t_p1 + t_p2 + t_p3 + t_p4) / float(len(y) + e), 4)
    accuracy_1 = round(float(t_p1 + t_n1) / float(t_p1 + t_n1 + f_n1 + f_p1 + e), 4)
    precision_1 = round(float(t_p1) / float(t_p1 + f_p1 + e), 4)
    recall_1 = round(float(t_p1) / float(t_p1 + f_n1 + e), 4)
    f1 = round(2 * precision_1 * recall_1 / (precision_1 + recall_1 + e), 4)

    accuracy_2 = round(float(t_p2 + t_n2) / float(t_p2 + t_n2 + f_n2 + f_p2 + e), 4)
    precision_2 = round(float(t_p2) / float(t_p2 + f_p2 + e), 4)
    recall_2 = round(float(t_p2) / float(t_p2 + f_n2 + e), 4)
    f2 = round(2 * precision_2 * recall_2 / (precision_2 + recall_2 + e), 4)

    accuracy_3 = round(float(t_p3 + t_n3) / float(t_p3 + t_n3 + f_n3 + f_p3 + e), 4)
    precision_3 = round(float(t_p3) / float(t_p3 + f_p3 + e), 4)
    recall_3 = round(float(t_p3) / float(t_p3 + f_n3 + e), 4)
    f3 = round(2 * precision_3 * recall_3 / (precision_3 + recall_3 + e), 4)

    accuracy_4 = round(float(t_p4 + t_n4) / float(t_p4 + t_n4 + f_n4 + f_p4 + e), 4)
    precision_4 = round(float(t_p4) / float(t_p4 + f_p4 + e), 4)
    recall_4 = round(float(t_p4) / float(t_p4 + f_n4 + e), 4)
    f4 = round(2 * precision_4 * recall_4 / (precision_4 + recall_4 + e), 4)

    micro_f = round((f1 + f2 + f3 + f4) / 4, 5)
    rmse_all = round((rmse / len(y)) ** 0.5, 4)
    rmse_all_1 = round((rmse1 / len(y)) ** 0.5, 4)
    rmse_all_2 = round((rmse2 / len(y)) ** 0.5, 4)
    rmse_all_3 = round((rmse3 / len(y)) ** 0.5, 4)
    rmse_all_4 = round((rmse4 / len(y)) ** 0.5, 4)
    rmse_all_avg = round((rmse_all_1 + rmse_all_2 + rmse_all_3 + rmse_all_4) / 4, 4)
    return ['acc:', accuracy_all, 'Favg:', micro_f, rmse_all, rmse_all_avg,
            'C1:', accuracy_1, precision_1, recall_1, f1,
            'C2:', accuracy_2, precision_2, recall_2, f2,
            'C3:', accuracy_3, precision_3, recall_3, f3,
            'C4:', accuracy_4, precision_4, recall_4, f4]


def write2Predict_oneVSall(prediction, y, result_path):  # no. of time series
    f_out = open(result_path, 'w')
    for i in range(len(y)):
        f_out.write(str(prediction[i][0]) + "\n")
    f_out.close()


def write2Predict_4class(prediction, y, result_path):  # no. of time series
    f_out = open(result_path, 'w')
    for i in range(len(y)):
        data1 = str(y[i][0]) + ' ' + str(y[i][1]) + ' ' + str(y[i][2]) + ' ' + str(y[i][3])
        data2 = str(prediction[i][0]) + ' ' + str(prediction[i][1]) + ' ' + str(prediction[i][2]) + ' ' + str(
            prediction[i][3])
        f_out.write(data1 + '\t' + data2 + "\n")
    f_out.close()
