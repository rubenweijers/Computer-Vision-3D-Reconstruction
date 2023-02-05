import os
import glm
import json
import numpy as np
from OpenGL.GL import *
from engine.renderable.mesh import Mesh


class Model:
    def __init__(self, path, rotation=(0, 0, 0)):
        self.meshes = []
        if not os.path.exists(path):
            raise RuntimeError(f'Model source file {path} does not exists.')
        self.path = path
        self.model = glm.mat4()
        self.rotation = glm.mat4(1)
        self.rotation = glm.rotate(self.rotation, rotation[0] * np.pi / 180, [1, 0, 0])
        self.rotation = glm.rotate(self.rotation, rotation[1] * np.pi / 180, [0, 1, 0])
        self.rotation = glm.rotate(self.rotation, rotation[2] * np.pi / 180, [0, 0, 1])
        data = self._load_get_data()
        for meshData in data['meshes']:
            self.meshes.append(Mesh(meshData))

    def _load_get_data(self):
        with open(self.path) as file:
            data = json.load(file)
        return data

    def set_multiple_positions(self, positions):
        for mesh in self.meshes:
            mesh.set_multiple_positions(positions)

    def draw(self, program):
        program.use()
        program.setMat4('model', self.model)
        for mesh in self.meshes:
            mesh.draw()

    def draw_multiple(self, program):
        program.use()
        program.setMat4('model', self.model)
        program.setMat4('rotation', self.rotation)
        for mesh in self.meshes:
            mesh.draw_multiple()

    def __del__(self):
        self.delete()

    def delete(self):
        self.meshes.clear()
