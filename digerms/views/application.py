# coding: utf-8
from pyglet import app, clock
from pyglet.window import key, Window, get_platform

from pyglet.gl import (
    glBlendFunc, glClear, glClearColor, glEnable,
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINE_SMOOTH, GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA)

class Application(object):
    # Initialize the view
    def __init__(self, fullscreen=False,
                 screen=None,
                 width=None, height=None,
                 visible=False, vsync=False, fps=60,
                 show_fps=False):
        platform = get_platform()
        display = platform.get_default_display()
        screens = display.get_screens()
        screen=screens[screen]
        self.window = Window(fullscreen=fullscreen, screen=screen,
            width=width, height=height, visible=visible,
            vsync=vsync
        )
        # glClearColor(0., 0., 0., 0.0)
        # glEnable(GL_BLEND)
        # glEnable(GL_LINE_SMOOTH)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.clockDisplay = clock.ClockDisplay(color=(1., 1., 1., .8)) if show_fps \
                            else None
        self.paused = False
        clock.set_fps_limit(fps)

        self.window.on_draw = self.on_draw
        self.window.on_key_press = self.on_key_press
        self.window.push_handlers(on_key_press=self.on_key_press)

    def set_mode(self, mode, interval):
        self.mode = mode
        clock.schedule_interval(self.simulate, interval)

    def on_draw(self):
        # glClear(GL_COLOR_BUFFER_BIT)
        self.window.clear()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.mode.draw()
        if self.clockDisplay:
            self.clockDisplay.draw()

    def on_mode_change(self, Mode):
        self.mode = Mode(self.window)
        self.mode.on_mode_change = self.on_mode_change

    def simulate(self, dt):
        self.mode.simulate(dt)

    def on_key_press(self, symbol, modifiers):
        self.mode.on_key_press(symbol, modifiers)

    def run(self):
        self.window.set_visible()
        app.run()
