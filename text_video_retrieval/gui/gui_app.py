import argparse
from pathlib import Path
import sys

from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut 
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .gui_worker import BackendWorker
from .video_player import VideoCard


class DatabaseDialog(QDialog):
    """Browse exact database keys without touching the video files."""

    remove_requested = Signal(object, object)

    def __init__(self, database_path, parent):
        super().__init__(parent)
        self.setWindowTitle("Videos in database")
        self.setModal(True)
        self.resize(800, 550)
        self.busy = False
        layout = QVBoxLayout(self)
        label = QLabel(f"Database: {database_path}")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(QLabel(
            "Select a video or directory to remove its database entries.\n"
            "Files on disk are never deleted. Directory removal affects only "
            "the videos directly listed under it."
        ))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Directory / video", "Videos"])
        self.tree.setColumnWidth(0, 620)
        self.tree.itemSelectionChanged.connect(self.update_controls)
        layout.addWidget(self.tree)
        self.count = QLabel()
        layout.addWidget(self.count)
        buttons = QHBoxLayout()
        self.remove_button = QPushButton("Remove selected from database…")
        self.remove_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.remove_button)
        buttons.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def set_rows(self, rows):
        self.tree.clear()
        directories = {}
        for directory, filename in rows:
            if directory not in directories:
                group = QTreeWidgetItem(self.tree, [directory])
                group.setData(0, Qt.ItemDataRole.UserRole, (directory, None))
                directories[directory] = group
            group = directories[directory]
            item = QTreeWidgetItem(group, [filename])
            item.setData(0, Qt.ItemDataRole.UserRole, (directory, filename))
        for group in directories.values():
            group.setText(1, str(group.childCount()))
        self.tree.expandAll()
        self.count.setText(f"{len(rows)} videos in {len(directories)} directories.")
        self.update_controls()

    def set_busy(self, busy):
        self.busy = busy
        self.update_controls()

    def update_controls(self):
        self.tree.setEnabled(not self.busy)
        self.remove_button.setEnabled(
            not self.busy and bool(self.tree.selectedItems())
        )
        self.close_button.setEnabled(not self.busy)

    def reject(self):
        if not self.busy:
            super().reject()

    def closeEvent(self, event):
        if self.busy:
            event.ignore()
        else:
            super().closeEvent(event)

    def remove_selected(self):
        selected = self.tree.selectedItems()
        if self.busy or not selected:
            return
        item = selected[0]
        directory, filename = item.data(0, Qt.ItemDataRole.UserRole)
        count = item.childCount() if filename is None else 1
        target = directory if filename is None else str(Path(directory) / filename)
        box = QMessageBox(self)
        box.setWindowTitle("Remove database entries?")
        box.setIcon(QMessageBox.Icon.Question)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(
            f"Remove {count} video(s) from this database?\n\n{target}\n\n"
            "The files will remain on disk."
        )
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() == QMessageBox.StandardButton.Yes:
            self.remove_requested.emit(
                [] if filename is None else [(directory, filename)],
                [directory] if filename is None else [],
            )


class RetrievalWindow(QMainWindow):
    requested = Signal(str, object)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Text to video retrieval")
        self.resize(1000, 900)
        self.busy = False
        self.loaded = {"vision": False, "text": False}
        self.cards = []
        self.database_dialog = None
        # One application-wide Esc shortcut (works while a video is
        # fullscreen in its own window).
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.esc_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.esc_shortcut.activated.connect(self.exit_fullscreen)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top_box = QHBoxLayout()
        self.model_buttons = {}
        for kind in ("vision", "text"):
            button = QPushButton()
            button.clicked.connect(
                lambda checked=False, name=kind: self.toggle_encoder(name)
            )
            self.model_buttons[kind] = button
            top_box.addWidget(button)
        self.dense_checkbox = QCheckBox("Dense video sampling")
        self.dense_checkbox.setToolTip(
            "Checked: frames are sampled more densely and the dense "
            "database is used for indexing and search."
        )
        top_box.addWidget(self.dense_checkbox)
        layout.addLayout(top_box)

        directory_group = QGroupBox("Add local videos to the database")
        directory_layout = QHBoxLayout(directory_group)
        self.directory = QLineEdit()
        self.directory.setPlaceholderText(
            "Path to a video file or a directory containing videos"
        )
        directory_layout.addWidget(self.directory, 1)
        self.browse_button = QPushButton("Directory…")
        self.browse_button.clicked.connect(self.browse)
        directory_layout.addWidget(self.browse_button)
        self.file_button = QPushButton("File…")
        self.file_button.clicked.connect(self.browse_file)
        directory_layout.addWidget(self.file_button)
        self.index_button = QPushButton("Index videos")
        self.index_button.clicked.connect(self.index_videos)
        directory_layout.addWidget(self.index_button)
        layout.addWidget(directory_group)
        self.database_button = QPushButton("List / manage database videos…")
        self.database_button.clicked.connect(self.browse_database)
        layout.addWidget(self.database_button)

        search_group = QGroupBox("Search")
        search_layout = QHBoxLayout(search_group)
        self.query = QLineEdit()
        self.query.setPlaceholderText(
            "For example: A cat playing"
        )
        self.query.returnPressed.connect(self.search)
        search_layout.addWidget(self.query, 1)
        search_layout.addWidget(QLabel("Top k:"))
        self.k = QSpinBox()
        self.k.setRange(1, 10000)
        self.k.setValue(3)
        search_layout.addWidget(self.k)
        search_layout.addWidget(QLabel("Min similarity:"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(args.threshold)
        search_layout.addWidget(self.threshold)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search)
        search_layout.addWidget(self.search_button)
        layout.addWidget(search_group)

        self.status = QLabel(
            "Load the vision encoder to index videos, or the text encoder "
            "to search an existing database."
        )
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        results = QWidget()
        self.results_layout = QVBoxLayout(results)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(results)
        layout.addWidget(scroll, 1)

        # This thread only relays messages to the separate backend process.
        self.thread = QThread(self)
        self.worker = BackendWorker(args)
        self.worker.moveToThread(self.thread)
        queued = Qt.ConnectionType.QueuedConnection
        self.requested.connect(self.worker.execute, queued)
        self.worker.succeeded.connect(self.job_succeeded, queued)
        self.worker.failed.connect(self.job_failed, queued)
        self.worker.progress.connect(self.show_progress, queued)
        self.worker.reset.connect(self.backend_reset, queued)
        self.thread.finished.connect(self.worker.shutdown)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self.update_controls()
        # Load the text encoder as soon as the window is up.
        if args.autoload_text:
            QTimer.singleShot(0, lambda: self.start_job("text", None))

    @Slot(str)
    def show_progress(self, message):
        self.status.setText(message)

    @Slot()
    def backend_reset(self):
        self.loaded = {"vision": False, "text": False}
        self.update_controls()

    def update_controls(self):
        for kind, button in self.model_buttons.items():
            ready = self.loaded[kind]
            button.setText(
                f"Unload {kind} encoder"
                if ready
                else f"Load {kind} encoder"
            )
            color = "#278447" if ready else "#727272"
            button.setStyleSheet(
                f"QPushButton {{ background-color: {color}; "
                "color: white; padding: 10px; border-radius: 4px; }"
            )
            button.setEnabled(not self.busy)
        self.index_button.setEnabled(not self.busy and self.loaded["vision"])
        self.search_button.setEnabled(not self.busy and self.loaded["text"])
        for widget in (
            self.directory, self.browse_button, self.query, self.k,
            self.threshold, self.file_button, self.database_button,
        ):
            widget.setEnabled(not self.busy)
        self.dense_checkbox.setEnabled(not self.busy)
        self.progress.setVisible(self.busy)
        if self.database_dialog is not None:
            self.database_dialog.set_busy(self.busy)

    def toggle_encoder(self, kind):
        action = f"unload_{kind}" if self.loaded[kind] else kind
        payload = None
        if action == "vision":
            payload = self.dense_checkbox.isChecked()
        self.start_job(action, payload)

    def database_name(self):
        return Path(
            self.args.dense_database
            if self.dense_checkbox.isChecked()
            else self.args.database
        ).name

    def browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose video directory", self.directory.text()
        )
        if path:
            self.directory.setText(path)

    def index_videos(self):
        path = self.directory.text().strip()
        source = Path(path).expanduser()
        if not path or not (source.is_dir() or source.is_file()):
            QMessageBox.warning(
                self,
                "Video source",
                "Choose an existing video file or directory containing videos.",
            )
            return
        self.start_job("index", (path, self.dense_checkbox.isChecked()))

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a video file", self.directory.text(),
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)",
        )
        if path:
            self.directory.setText(path)

    def browse_database(self):
        self.start_job("list_videos", self.dense_checkbox.isChecked())

    def remove_database_entries(self, videos, directories):
        self.start_job(
            "delete_entries",
            (videos, directories, self.dense_checkbox.isChecked()),
        )

    def search(self):
        if self.busy or not self.loaded["text"]:
            return
        query = self.query.text().strip()
        if not query:
            QMessageBox.warning(self, "Query", "Enter a text query.")
            return
        self.threshold_used = self.threshold.value()
        self.start_job(
            "search",
            (
                query,
                self.k.value(),
                self.threshold_used,
                self.dense_checkbox.isChecked(),
            ),
        )

    def start_job(self, action, payload):
        if self.busy:
            return
        self.busy = True
        self.update_controls()
        messages = {
            "vision": "Loading vision encoder…",
            "text": "Loading text encoder…",
            "unload_vision": "Unloading vision encoder…",
            "unload_text": "Unloading text encoder…",
            "index": "Encoding videos and updating the database…",
            "search": "Searching…",
            "list_videos": "Listing database videos…",
            "delete_entries": "Removing database entries…",
        }
        self.status.setText(messages[action])
        kind = action.removeprefix("unload_")
        if kind in self.model_buttons:
            self.model_buttons[kind].setText(messages[action])
        self.requested.emit(action, payload)

    @Slot(str, object)
    def job_succeeded(self, action, result):
        self.busy = False
        if action in self.loaded:
            self.loaded[action] = True
            self.status.setText(f"{action.capitalize()} encoder loaded.")
        elif action in ("unload_vision", "unload_text"):
            kind = action.removeprefix("unload_")
            self.loaded[kind] = False
            message = f"{kind.capitalize()} encoder unloaded."
            if result:
                message += f" {result}"
            self.status.setText(message)
        elif action == "index":
            self.status.setText(
                f"Added or updated {result} videos."
                if result
                else "No videos were returned by the encoder."
            )
        elif action == "list_videos":
            if self.database_dialog is not None:
                self.database_dialog.deleteLater()
            path = (
                self.args.dense_database if self.dense_checkbox.isChecked()
                else self.args.database
            )
            self.database_dialog = DatabaseDialog(
                str(Path(path).expanduser().resolve()), self
            )
            self.database_dialog.remove_requested.connect(self.remove_database_entries)
            self.database_dialog.set_rows(result)
            self.database_dialog.show()
            self.status.setText(f"Listed {len(result)} videos in {self.database_name()}.")
        elif action == "delete_entries":
            removed, rows = result
            self.database_dialog.set_rows(rows)
            self.clear_results()
            self.status.setText(
                f"Removed {removed} database entries. Files on disk are unchanged."
            )
        elif action == "search":
            matches, missing, below = result
            threshold = self.threshold_used
            self.clear_results()
            for rank, (path, score, thumb) in enumerate(matches, 1):
                card = VideoCard(rank, path, score, thumb)
                card.play_requested.connect(self.pause_other_videos)
                self.cards.append(card)
                self.results_layout.addWidget(card)
            if matches:
                message = (
                    f"Showing {len(matches)} videos with similarity "
                    f"≥ {threshold:.2f}."
                )
            else:
                message = (
                    f"No match: no video reached similarity "
                    f"{threshold:.2f}. Lower the threshold or index "
                    "more videos."
                )
            if below and matches:
                message += f" {below} other videos were below the threshold."
            if missing:
                message += f" Skipped {missing} missing files."
            self.status.setText(message)
        self.update_controls()

    @Slot(str, str)
    def job_failed(self, action, message):
        self.busy = False
        self.update_controls()
        self.status.setText(f"{action.capitalize()} failed. You can retry.")
        box = QMessageBox(self)
        box.setWindowTitle("Operation failed")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(message)
        box.exec()

    def exit_fullscreen(self):
        for card in self.cards:
            card.exit_fullscreen()

    def pause_other_videos(self, selected):
        for card in self.cards:
            if card is not selected:
                card.pause()

    def clear_results(self):
        for card in self.cards:
            card.release()
            self.results_layout.removeWidget(card)
            card.deleteLater()
        self.cards.clear()

    def closeEvent(self, event):
        # The relay polls this thread-safe flag even if inference is stuck.
        self.worker.stopping.set()
        self.clear_results()
        self.thread.quit()
        self.thread.wait()
        event.accept()


def parse_args():
    ROOT = Path("/")
    INTERNVIDEO_PATH = ROOT / "InternVideo"
    DATABASE_PATH = ROOT / "app-data"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-autoload-text",
        dest="autoload_text",
        action="store_false",
        help="Do not load the text encoder automatically at startup",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.32,
        help="Minimum cosine similarity for a video to count as a match",
    )
    parser.add_argument(
        "--database",
        default=str(
            DATABASE_PATH / "videos.sqlite"
        ),
    )
    parser.add_argument(
        "--dense-database",
        default=str(
            DATABASE_PATH / "videos_dense.sqlite"
        ),
        help="Database used when 'Dense video sampling' is checked",
    )
    parser.add_argument(
        "--vision-config",
        default=str(
            INTERNVIDEO_PATH / "internvideo2_config.py"
        ),
    )
    parser.add_argument(
        "--text-config",
        default=str(
            INTERNVIDEO_PATH / "internvideo2_config.py"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv[:1])
    window = RetrievalWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
