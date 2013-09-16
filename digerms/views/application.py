# coding: utf-8
from pyglet import app, clock
from pyglet.window import key, Window

from pyglet.gl import (
    glBlendFunc, glClear, glClearColor, glEnable,
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINE_SMOOTH, GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA)

class Renderer(object):
    def __init__(self, show_fps=False):
        glClearColor(0., 0., 0., 0.0)
        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.clockDisplay = clock.ClockDisplay(color=(1., 1., 1., .8)) if show_fps \
                            else None

    def on_draw(self, scene, win_width, win_height):
        glClear(GL_COLOR_BUFFER_BIT)
        scene.get_batch().draw()
        if self.clockDisplay:
            self.clockDisplay.draw()

class Application(object):
    # Initialize the view
    def __init__(self, fullscreen=False,
                 width=None, height=None,
                 visible=False, vsync=False, fps=60,
                 show_fps=False):
        self.window = Window(fullscreen=fullscreen,
            width=width, height=height, visible=visible, vsync=vsync
        )
        self.renderer = Renderer(show_fps)
        self.paused = False
        clock.set_fps_limit(fps)

    def update(self, dt):
        if not self.paused:
            self.scene.update(dt)

    def set_scene(self, scene, interval):
        self.scene = scene
        clock.schedule_interval(self.update, interval)
        self.window.on_draw = lambda: self.renderer.on_draw(self.scene, self.window.width, self.window.height)

    def toggle_pause(self):
        self.paused = not self.paused

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            app.exit()
        elif symbol == key.RIGHT:
            if self.paused:
                self.scene.update(0)
            else:
                self.toggle_pause()
        elif symbol == key.SPACE:
            self.toggle_pause()

    def run(self):
        self.window.set_visible()
        self.window.on_key_press = self.on_key_press
        app.run()
