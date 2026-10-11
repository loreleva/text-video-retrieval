"""YouTube-style video player widgets and the result card."""

from pathlib import Path

from PySide6.QtCore import QPoint, QSizeF, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPixmap, QPolygon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (
    QFrame, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget,
)

VIDEO_HEIGHT = 320  # fixed height of the preview/video area in each card


# ---------------------------------------------------------------- player UI

def format_time(ms):
    seconds = max(0, int(ms // 1000))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class ClickSlider(QSlider):
    """Slider that jumps to the clicked position."""

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > 0:
            value = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                int(event.position().x()), max(1, self.width()),
            )
            self.setValue(value)
            self.sliderMoved.emit(value)
        super().mousePressEvent(event)


BAR_STYLE = """
#bar { background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
       stop:0 rgba(0,0,0,0), stop:1 rgba(0,0,0,210)); }
#bar QPushButton { background: transparent; border: none; padding: 4px; }
#bar QPushButton:hover { background: rgba(255,255,255,45);
       border-radius: 4px; }
#bar QLabel { color: white; background: transparent; }
#bar QSlider::groove:horizontal { height: 4px; border-radius: 2px;
       background: rgba(255,255,255,90); }
#bar QSlider::sub-page:horizontal { background: #ff2d2d; border-radius: 2px; }
#bar QSlider::handle:horizontal { background: #ff2d2d; width: 12px;
       height: 12px; margin: -4px 0; border-radius: 6px; }
"""


class PlayerView(QGraphicsView):
    """YouTube-style player: video (or thumbnail) with overlaid controls.

    Media objects are only created on the first Play, so showing search
    results never touches the decoders. Fullscreen is handled by the card
    (re-parenting this plain widget), which avoids QVideoWidget's problem
    of not returning to its original place.
    """

    fullscreen_requested = Signal()
    play_started = Signal()
    error_occurred = Signal(str)

    def __init__(self, path, thumbnail=None):
        super().__init__()
        self.path = path
        self.player = None
        self.audio = None
        self.video_item = None
        self.is_fullscreen = False
        self.showing_video = False
        self.volume_value = 70
        self.muted = False
        self._ignore_release = False

        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("black"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMinimumSize(1, 1)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # Thumbnail (or a text placeholder) shown before the first Play.
        self.thumb_pixmap = None
        self.thumb_item = self.scene().addPixmap(QPixmap())
        self.placeholder = self.scene().addSimpleText(
            "No preview available (install opencv-python-headless)\n"
            "Click to play"
        )
        self.placeholder.setBrush(QColor("#bbbbbb"))
        if thumbnail:
            pixmap = QPixmap()
            if pixmap.loadFromData(thumbnail):
                self.thumb_pixmap = pixmap
                self.placeholder.hide()
        if self.thumb_pixmap is None:
            self.thumb_item.hide()

        self.click_timer = QTimer(self)
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.toggle_play)
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide_controls)
        self.build_bar()
        self.layout_scene()

    # ------------------------------------------------------------ controls
    def build_bar(self):
        style = self.style()
        icon = style.standardIcon
        SP = QStyle.StandardPixmap
        self.icons = {
            "play": icon(SP.SP_MediaPlay),
            "pause": icon(SP.SP_MediaPause),
            "stop": icon(SP.SP_MediaStop),
            "volume": icon(SP.SP_MediaVolume),
            "muted": icon(SP.SP_MediaVolumeMuted),
            "enter": icon(SP.SP_TitleBarMaxButton),
            "exit": icon(SP.SP_TitleBarNormalButton),
        }
        self.bar = QWidget(self)
        self.bar.setObjectName("bar")
        self.bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.bar.setStyleSheet(BAR_STYLE)
        column = QVBoxLayout(self.bar)
        column.setContentsMargins(10, 18, 10, 6)
        column.setSpacing(2)

        self.seek = ClickSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderMoved.connect(self.seek_to)
        column.addWidget(self.seek)

        row = QHBoxLayout()
        self.play_button = QPushButton()
        self.play_button.setIcon(self.icons["play"])
        self.play_button.clicked.connect(self.toggle_play)
        row.addWidget(self.play_button)
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.icons["stop"])
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        self.time_label = QLabel("00:00 / 00:00")
        row.addWidget(self.time_label)
        row.addStretch(1)
        self.mute_button = QPushButton()
        self.mute_button.setIcon(self.icons["volume"])
        self.mute_button.clicked.connect(self.toggle_mute)
        row.addWidget(self.mute_button)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(self.volume_value)
        self.volume.setFixedWidth(80)
        self.volume.valueChanged.connect(self.set_volume)
        row.addWidget(self.volume)
        self.fs_button = QPushButton()
        self.fs_button.setIcon(self.icons["enter"])
        self.fs_button.clicked.connect(self.fullscreen_requested)
        row.addWidget(self.fs_button)
        column.addLayout(row)
        self.bar.raise_()

    def set_fullscreen_state(self, fullscreen):
        self.is_fullscreen = fullscreen
        self.fs_button.setIcon(self.icons["exit" if fullscreen else "enter"])
        self.show_controls()

    def show_controls(self):
        self.bar.show()
        self.bar.raise_()
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        if self.is_playing():
            self.hide_timer.start(2500)

    def hide_controls(self):
        if self.is_playing() and not self.bar.underMouse():
            self.bar.hide()
            if self.is_fullscreen:
                self.viewport().setCursor(Qt.CursorShape.BlankCursor)

    # -------------------------------------------------------------- layout
    def layout_scene(self):
        size = self.viewport().size()
        w, h = max(1, size.width()), max(1, size.height())
        self.scene().setSceneRect(0, 0, w, h)
        if self.thumb_pixmap is not None:
            scaled = self.thumb_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.thumb_item.setPixmap(scaled)
            self.thumb_item.setPos(
                (w - scaled.width()) / 2, (h - scaled.height()) / 2
            )
        else:
            rect = self.placeholder.boundingRect()
            self.placeholder.setPos(
                (w - rect.width()) / 2, (h - rect.height()) / 2
            )
        if self.video_item is not None:
            self.video_item.setSize(QSizeF(w, h))
        bar_h = self.bar.sizeHint().height()
        self.bar.setGeometry(0, self.height() - bar_h, self.width(), bar_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.layout_scene()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.showing_video:
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, r = self.width() // 2, self.height() // 2, 30
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(cx, cy), r, r)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawPolygon(
            QPolygon([
                QPoint(cx - 9, cy - 14),
                QPoint(cx - 9, cy + 14),
                QPoint(cx + 15, cy),
            ])
        )
        painter.end()

    # ------------------------------------------------------------ playback
    def ensure_player(self):
        if self.player is not None:
            return
        self.video_item = QGraphicsVideoItem()
        self.video_item.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.scene().addItem(self.video_item)
        self.video_item.hide()
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(self.volume_value / 100)
        self.audio.setMuted(self.muted)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_item)
        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        self.player.playbackStateChanged.connect(self.update_state)
        self.player.errorOccurred.connect(
            lambda error, message: self.error_occurred.emit(
                f"Playback error: {message}"
            )
        )
        self.player.setSource(QUrl.fromLocalFile(self.path))
        self.layout_scene()

    def is_playing(self):
        return (
            self.player is not None
            and self.player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        )

    def show_video(self, visible):
        self.showing_video = visible
        self.thumb_item.setVisible(not visible and self.thumb_pixmap is not None)
        self.placeholder.setVisible(
            not visible and self.thumb_pixmap is None
        )
        if self.video_item is not None:
            self.video_item.setVisible(visible)
        self.viewport().update()

    @Slot()
    def toggle_play(self):
        self.ensure_player()
        if self.is_playing():
            self.player.pause()
            return
        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)
        self.play_started.emit()
        self.show_video(True)
        self.player.play()

    def stop(self):
        if self.player is not None:
            self.player.stop()
            self.show_video(False)
        self.show_controls()

    def pause(self):
        if self.player is not None:
            self.player.pause()

    def release(self):
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())

    def seek_to(self, ms):
        if self.player is not None:
            self.player.setPosition(ms)

    def toggle_mute(self):
        self.muted = not self.muted
        if self.audio is not None:
            self.audio.setMuted(self.muted)
        self.update_volume_icon()

    def set_volume(self, value):
        self.volume_value = value
        if value > 0 and self.muted:
            self.muted = False
            if self.audio is not None:
                self.audio.setMuted(False)
        if self.audio is not None:
            self.audio.setVolume(value / 100)
        self.update_volume_icon()

    def update_volume_icon(self):
        silent = self.muted or self.volume_value == 0
        self.mute_button.setIcon(self.icons["muted" if silent else "volume"])

    def update_duration(self, ms):
        self.seek.setRange(0, ms)
        self.update_time(self.player.position(), ms)

    def update_position(self, ms):
        if not self.seek.isSliderDown():
            self.seek.setValue(ms)
        self.update_time(ms, self.player.duration())

    def update_time(self, position, duration):
        self.time_label.setText(
            f"{format_time(position)} / {format_time(duration)}"
        )

    def update_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setIcon(self.icons["pause" if playing else "play"])
        if playing:
            self.hide_timer.start(2500)
        else:
            self.hide_timer.stop()
            self.bar.show()

    # --------------------------------------------------------------- input
    def mouseMoveEvent(self, event):
        self.show_controls()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.is_playing():
            self.hide_timer.start(800)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._ignore_release:
            self._ignore_release = False
        elif event.button() == Qt.MouseButton.LeftButton:
            # Wait briefly so a double click doesn't also toggle play.
            self.click_timer.start(250)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.click_timer.stop()
        self._ignore_release = True
        self.fullscreen_requested.emit()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self.is_fullscreen:
                self.fullscreen_requested.emit()
        elif key == Qt.Key.Key_F:
            self.fullscreen_requested.emit()
        elif key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_M:
            self.toggle_mute()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            if self.player is not None:
                step = 5000 if key == Qt.Key.Key_Right else -5000
                self.player.setPosition(
                    max(0, self.player.position() + step)
                )
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # Closing the fullscreen window returns the video to its card.
        if self.is_fullscreen:
            event.ignore()
            self.fullscreen_requested.emit()
        else:
            super().closeEvent(event)


class VideoCard(QGroupBox):
    play_requested = Signal(object)

    def __init__(self, rank, path, score, thumbnail=None):
        super().__init__(
            f"{rank}. {Path(path).name} | cosine similarity: {score:.3f}"
        )
        layout = QVBoxLayout(self)
        label = QLabel(path)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(label)

        self.path = str(path)
        # Fixed-height slot: the video is letterboxed inside it.
        self.stage = QWidget()
        self.stage.setFixedHeight(VIDEO_HEIGHT)
        self.stage.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        stage_layout = QVBoxLayout(self.stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        self.view = PlayerView(self.path, thumbnail)
        stage_layout.addWidget(self.view)
        self.view.play_started.connect(
            lambda: self.play_requested.emit(self)
        )
        self.view.fullscreen_requested.connect(self.toggle_fullscreen)
        self.view.error_occurred.connect(self.show_error)
        layout.addWidget(self.stage)

        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.error)

    def toggle_fullscreen(self):
        view = self.view
        if not view.is_fullscreen:
            self.stage.layout().removeWidget(view)
            view.setParent(None, Qt.WindowType.Window)
            view.set_fullscreen_state(True)
            view.showFullScreen()
            view.activateWindow()
            view.setFocus()
        else:
            view.showNormal()
            view.set_fullscreen_state(False)
            view.setParent(self.stage, Qt.WindowType.Widget)
            self.stage.layout().addWidget(view)
            view.show()
            view.setFocus()

    def exit_fullscreen(self):
        if self.view.is_fullscreen:
            self.toggle_fullscreen()

    def pause(self):
        self.view.pause()

    def release(self):
        self.exit_fullscreen()
        self.view.release()

    def show_error(self, message):
        self.error.setText(message)
