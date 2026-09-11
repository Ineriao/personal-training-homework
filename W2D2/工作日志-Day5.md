# 工作日志

## 日期

Day5

## 项目主题

手势识别

## 今日完成

1. 使用 HaGRID Sample 30k 384p 数据集整理五类静态手势图片。
2. 将 JSON 边界框标注转换为 YOLO TXT 格式。
3. 完成训练集、验证集和测试集划分，数量分别为 2100、600 和 300 张。
4. 使用 `data.yaml` 配置类别和数据集路径。
5. 使用 YOLO11n 完成 100 轮迁移学习训练，训练尺寸为 `320`，批大小为 `64`。
6. 完成测试集评估和 300 张预测结果图生成。

## 测试集指标

| 指标 | 结果 |
| --- | ---: |
| Precision | 0.974238 |
| Recall | 0.976060 |
| mAP50 | 0.991722 |
| mAP50-95 | 0.858553 |

## 今日产物

- `convert_hagrid.py`
- `prepare_dataset.py`
- `validate_dataset.py`
- `data.yaml`
- `train_gesture_model.py`
- `evaluate_gesture_model.py`
- `predict_gestures.py`
- `dataset`
- `runs`
- `results`
