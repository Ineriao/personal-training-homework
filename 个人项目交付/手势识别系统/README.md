# 静态手势识别系统

## 1. 项目简介

本项目使用 YOLO11n 对图片中的手部区域进行目标检测，识别五类静态手势：`fist`、`like`、`ok`、`palm`、`peace`。系统提供 PyQt5 图形界面，可以打开一张图片，调用训练好的模型完成识别，并显示检测框、手势类别、置信度和目标数量。

## 2. 交付内容

```text
手势识别系统/
├─ app/                         图形界面程序
├─ scripts/                     预测和数据检查脚本
├─ training/                    数据转换和训练源代码
├─ models/best.pt               训练得到的最佳模型
├─ config/                      数据集和运行配置示例
├─ samples/demo.jpg             代表性输入图片
├─ outputs/metrics/             训练曲线和评价图
├─ docs/                        项目文档、测试和发布说明
├─ release/                     启动脚本
└─ requirements.txt             Python 依赖
```

交付包不包含原始数据集压缩包、完整训练集、训练批次中间图片和完整预测结果目录。它们属于数据处理或训练过程材料，保留在每日作业目录中。

## 3. 环境安装

建议使用 Python 3.11 虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

如果需要使用 CUDA，应按照本机 NVIDIA 驱动对应的 PyTorch 官方安装命令安装 CUDA 版 PyTorch，再安装其余依赖。

## 4. 启动界面

在项目根目录执行：

```bash
python app/gesture_monitor.py
```

也可以双击 `release/启动手势识别.bat`。启动后点击“打开图片”，选择手势图片，再点击“开始识别”。

## 5. 命令行预测

```bash
python scripts/predict_gestures.py --source samples/demo.jpg --model models/best.pt
```

预测结果默认保存到 `outputs/predictions/gesture_predict`。

## 6. 训练结果

数据集包含五类手势，共 3000 张图片：训练集 2100 张、验证集 600 张、测试集 300 张。模型使用 YOLO11n 迁移学习训练 100 轮，输入尺寸为 `320`，批大小为 `64`。

独立测试集记录结果：

| 指标 | 结果 |
| --- | ---: |
| Precision | 0.974238 |
| Recall | 0.976060 |
| mAP50 | 0.991722 |
| mAP50-95 | 0.858553 |

## 7. 文档索引

- [项目计划](docs/项目计划.md)
- [需求分析](docs/需求分析.md)
- [软件设计](docs/软件设计.md)
- [数据集与标注说明](docs/数据集与标注说明.md)
- [模型训练记录](docs/模型训练记录.md)
- [测试用例与报告](docs/测试用例与报告.md)
- [软件发布说明](docs/软件发布说明.md)
- [工作日志索引](docs/工作日志索引.md)
- [交付件清单](docs/交付件清单.md)

