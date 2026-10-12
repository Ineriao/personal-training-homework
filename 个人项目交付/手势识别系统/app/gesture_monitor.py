import os
import sys
from collections import Counter
from pathlib import Path

import PyQt5


QT_PLUGIN_DIR = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
os.environ["QT_PLUGIN_PATH"] = str(QT_PLUGIN_DIR)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLUGIN_DIR / "platforms")
os.environ["QT_QPA_PLATFORM"] = "windows"

import cv2
import torch
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = PROJECT_DIR / "models/best.pt"
IMAGE_FILTER = "图片文件 (*.jpg *.jpeg *.png *.bmp)"


class InferenceWorker(QObject):
    completed = pyqtSignal(object, list)
    failed = pyqtSignal(str)

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.model_path = model_path
        self.model = None

    @pyqtSlot(str, float)
    def predict(self, image_path: str, confidence: float) -> None:
        try:
            if self.model is None:
                self.model = YOLO(self.model_path)

            device = 0 if torch.cuda.is_available() else "cpu"
            result = self.model.predict(
                source=image_path,
                conf=confidence,
                imgsz=320,
                device=device,
                verbose=False,
            )[0]

            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "name": result.names[class_id],
                        "confidence": float(box.conf.item()),
                    }
                )

            rendered = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            self.completed.emit(rendered, detections)
        except Exception as exc:
            self.failed.emit(str(exc))


class GestureMonitor(QMainWindow):
    request_prediction = pyqtSignal(str, float)

    def __init__(self) -> None:
        super().__init__()
        self.image_path = ""
        self.current_pixmap = QPixmap()
        self.setWindowTitle("静态手势识别系统")
        self.resize(1120, 700)
        self.setMinimumSize(900, 580)
        self._build_ui()
        self._build_worker()
        self.statusBar().showMessage("请选择一张手势图片")

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        title = QLabel("静态手势识别")
        title.setObjectName("title")
        subtitle = QLabel("支持 fist、like、ok、palm、peace 五类手势")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(16)

        preview_group = QGroupBox("图像预览")
        preview_layout = QVBoxLayout(preview_group)
        self.image_label = QLabel("尚未选择图片")
        self.image_label.setObjectName("imagePreview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(560, 400)
        preview_layout.addWidget(self.image_label)
        content.addWidget(preview_group, 3)

        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)
        summary_grid = QGridLayout()
        summary_grid.addWidget(QLabel("目标数量"), 0, 0)
        self.count_value = QLabel("0")
        self.count_value.setObjectName("metricValue")
        summary_grid.addWidget(self.count_value, 0, 1)
        summary_grid.addWidget(QLabel("主要手势"), 1, 0)
        self.gesture_value = QLabel("--")
        self.gesture_value.setObjectName("metricValue")
        summary_grid.addWidget(self.gesture_value, 1, 1)
        result_layout.addLayout(summary_grid)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        result_layout.addWidget(line)

        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["手势类别", "置信度"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionMode(QTableWidget.NoSelection)
        result_layout.addWidget(self.result_table)
        content.addWidget(result_group, 2)
        layout.addLayout(content, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.open_button = QPushButton("打开图片")
        self.detect_button = QPushButton("开始识别")
        self.detect_button.setObjectName("primaryButton")
        self.clear_button = QPushButton("清空")
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)

        self.open_button.clicked.connect(self.open_image)
        self.detect_button.clicked.connect(self.start_prediction)
        self.clear_button.clicked.connect(self.clear_content)
        self.setCentralWidget(central)
        self._apply_style()

    def _build_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.worker = InferenceWorker(MODEL_PATH)
        self.worker.moveToThread(self.worker_thread)
        self.request_prediction.connect(self.worker.predict)
        self.worker.completed.connect(self.show_prediction)
        self.worker.failed.connect(self.show_error)
        self.worker_thread.start()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f6f8; }
            QWidget { color: #20252b; font-family: "Microsoft YaHei"; font-size: 14px; }
            QLabel#title { font-size: 26px; font-weight: 700; }
            QLabel#subtitle { color: #66717d; margin-bottom: 4px; }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d8dee5;
                border-radius: 6px;
                font-weight: 600;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QLabel#imagePreview {
                background: #171b20;
                color: #aeb7c2;
                border: 1px solid #2e343b;
            }
            QLabel#metricValue { color: #147d64; font-size: 18px; font-weight: 700; }
            QTableWidget { border: 1px solid #d8dee5; gridline-color: #e8ecef; }
            QHeaderView::section { background: #eef1f4; padding: 7px; border: 0; }
            QPushButton {
                min-width: 110px;
                min-height: 38px;
                padding: 0 12px;
                background: #ffffff;
                border: 1px solid #bdc6cf;
                border-radius: 5px;
            }
            QPushButton:hover { background: #edf2f5; }
            QPushButton:disabled { color: #9ca4ad; background: #e8ebee; }
            QPushButton#primaryButton { color: #ffffff; background: #147d64; border-color: #147d64; }
            QPushButton#primaryButton:hover { background: #106b55; }
            """
        )

    def open_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(self, "选择手势图片", str(BASE_DIR), IMAGE_FILTER)
        if not image_path:
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "读取失败", "无法读取所选图片。")
            return
        self.image_path = image_path
        self.current_pixmap = pixmap
        self._refresh_preview()
        self._clear_results()
        self.statusBar().showMessage(f"已打开：{Path(image_path).name}")

    def start_prediction(self) -> None:
        if not self.image_path:
            QMessageBox.information(self, "提示", "请先打开一张手势图片。")
            return
        if not MODEL_PATH.is_file():
            QMessageBox.warning(self, "模型不存在", f"未找到模型文件：\n{MODEL_PATH}")
            return
        self.detect_button.setEnabled(False)
        self.statusBar().showMessage("正在识别，请稍候...")
        self.request_prediction.emit(self.image_path, 0.25)

    @pyqtSlot(object, list)
    def show_prediction(self, image: object, detections: list) -> None:
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.current_pixmap = QPixmap.fromImage(qimage)
        self._refresh_preview()

        self.result_table.setRowCount(len(detections))
        for row, detection in enumerate(detections):
            self.result_table.setItem(row, 0, QTableWidgetItem(detection["name"]))
            self.result_table.setItem(row, 1, QTableWidgetItem(f"{detection['confidence']:.1%}"))

        counts = Counter(item["name"] for item in detections)
        self.count_value.setText(str(len(detections)))
        self.gesture_value.setText(counts.most_common(1)[0][0] if counts else "未检测到")
        self.detect_button.setEnabled(True)
        self.statusBar().showMessage("识别完成")

    @pyqtSlot(str)
    def show_error(self, message: str) -> None:
        self.detect_button.setEnabled(True)
        self.statusBar().showMessage("识别失败")
        QMessageBox.critical(self, "识别失败", message)

    def clear_content(self) -> None:
        self.image_path = ""
        self.current_pixmap = QPixmap()
        self.image_label.setText("尚未选择图片")
        self.image_label.setPixmap(QPixmap())
        self._clear_results()
        self.statusBar().showMessage("内容已清空")

    def _clear_results(self) -> None:
        self.result_table.setRowCount(0)
        self.count_value.setText("0")
        self.gesture_value.setText("--")

    def _refresh_preview(self) -> None:
        if self.current_pixmap.isNull():
            return
        size = self.image_label.size()
        self.image_label.setPixmap(
            self.current_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_preview()

    def closeEvent(self, event) -> None:
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = GestureMonitor()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
