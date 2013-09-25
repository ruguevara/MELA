#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pyglet
from pyglet.event import EventDispatcher
from pygarrayimage.arrayimage import ArrayInterfaceImage

#########################################
from pyglet.graphics import OrderedGroup


class Vec(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec(self.x * other, self.y * other)
        else:
            raise Exception('Vec * %s not implemented' % type(other))

    def __div__(self, other):
        return Vec(self.x / other, self.y / other)

    def __repr__(self):
        return '<Vec %-04.2f, %-04.2f>' % (self.x, self.y)

    def rotate(self, deg):
        rad = deg * (math.pi/180.0)
        return Vec(
            math.cos(rad)*self.x - math.sin(rad)*self.y,
            math.sin(rad)*self.x + math.cos(rad)*self.y,
        )

    def angle(self, other):
        rad = math.atan2(other.y, other.x) - math.atan2(self.y, self.x)
        deg = rad * (180.0/math.pi)
        if deg < -180.0:
            return 360 + deg
        else:
            return deg

    def __len__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def distance(self, other):
        return len(self - other)


class Box(object):
    def __init__(self, x=0, y=0, w=0, h=0):
        EventDispatcher.__init__(self)
        self.pos = Vec(x, y)
        self.width = w
        self.height = h

    @property
    def top(self):
        return self.pos.y + self.height

    @property
    def bottom(self):
        return self.pos.y

    @property
    def left(self):
        return self.pos.x

    @property
    def right(self):
        return self.pos.x + self.width


class GraphicsObject(EventDispatcher):
    def __init__(self, x=0, y=0):
        EventDispatcher.__init__(self)
        self.pos = Vec(x, y)

    def simulate(self, delta):
        pass

    def draw(self):
        raise NotImplementedError

    def on_key_press(self, symbol, modifiers):
        pass

    def remove(self):
        self.dispatch_event('on_remove', self)
    #
    # def __del__(self):
    #     self.remove()

    def add_to_batch(self, batch=None, parent=None):
        # должно всегда вызываться после созданя объекта,
        # при добавлении объекта в группу или нет
        pass


GraphicsObject.register_event_type('on_remove')

class SpriteObject(GraphicsObject):
    """
    sprite
    """
    def __init__(self, x=0, y=0, scale=1, usage="dynamic", **kwargs):
        super(SpriteObject, self).__init__(x, y)
        self.scale = scale
        self.usage = usage

    def get_image(self):
        raise NotImplementedError()

    def update(self):
        self.sprite._set_texture(self.get_image().get_texture())

    def draw(self):
        self.update()
        self.sprite.draw()

    def add_to_batch(self, batch=None, parent=None):
        # должно всегда вызываться после созданя объекта,
        # при добавлении объекта в группу или нет
        self.group = parent
        self.sprite = pyglet.sprite.Sprite(self.get_image(), self.pos.x, self.pos.y,
                                           batch=batch, group=self.group, usage=self.usage)
        self.sprite.scale = self.scale

class BitMapObject(SpriteObject):
    """
    Dynamic non-moving sprite
    """
    def __init__(self, x=0, y=0, scale=1, **kwargs):
        super(BitMapObject, self).__init__(x, y, scale, "stream")

    def get_data(self):
        raise NotImplementedError()

    def get_image(self):
        return ArrayInterfaceImage(self.get_data())


class Mode(EventDispatcher):
    def on_change(self, mode):
        pass

    def simulate(self, delta):
        pass

    def draw(self):
        pass

    def on_key_press(self, symbol, modifiers):
        pass

Mode.register_event_type('on_mode_change')

class Group(GraphicsObject):
    def __init__(self, x=0, y=0):
        super(Group, self).__init__(x, y)
        self.objects = []
        self.batch = None
        self.parent_group=OrderedGroup(0)

    def add(self, instance, group=None):
        instance.push_handlers(on_remove=self.on_remove)
        self.objects.append(instance)
        instance.add_to_batch(self.batch, parent=group)
        return instance

    def add_to_batch(self, batch=None, parent=None):
        """
        группу добавляют в другую группу
        """
        self.batch = batch
        self.parent_group=parent if parent is not None else OrderedGroup(0)
        super(Group, self).add_to_batch(batch, parent)

    def add_objects(self, instances):
        for group, inst in enumerate(instances):
            g = OrderedGroup(group, self.parent_group)
            self.add(inst, group=g)
        return instances

    def on_remove(self, instance):
        # если instance вызывает событие on_remove
        instance.remove_handlers(self)
        if instance in self.objects:
            self.objects.remove(instance)

    def draw(self):
        for obj in self.objects:
            obj.draw()

    def simulate(self, delta):
        for obj in self.objects:
            obj.simulate(delta)

    def update(self):
        """
        готовит данные перед пакетной отрисовкой
        """
        for obj in self.objects:
            obj.update()

    def on_key_press(self, symbol, modifiers):
        for obj in self.objects:
            obj.on_key_press(symbol, modifiers)

    def __iter__(self):
        return iter(self.objects)

Group.register_event_type('on_remove')


class BatchGroup(Group):
    def __init__(self, x=0, y=0):
        super(BatchGroup, self).__init__(x, y)
        self.batch = pyglet.graphics.Batch()

    def draw(self):
        for obj in self.objects:
            obj.update()
        self.batch.draw()
