import glm
import glfw
import math
from OpenGL.GL import *
from engine.base.program import get_linked_program
from engine.renderable.model import Model
from engine.buffer.texture import *
from engine.buffer.hdrbuffer import HDRBuffer
from engine.buffer.blurbuffer import BlurBuffer
from engine.effect.bloom import Bloom
from engine.generator import generate_voxel_positions, generate_grid, get_cam_pos
from engine.camera import Camera
from engine.config import config

cube, square, hdrbuffer, blurbuffer, lastPosX, lastPosY = None, None, None, None, None, None
firstTime = True
width, height = config['window_width'], config['window_height']
# camera = Camera(glm.vec3(-5, config['world_height']+5, -5))
camera = Camera(glm.vec3(0, 250, 0))


def draw_objs(obj, program, perspective, light_pos, texture, normal, specular, depth):
    program.use()
    program.setMat4('viewProject', perspective * camera.get_view_matrix())
    program.setVec3('viewPos', camera.position)
    program.setVec3('light_pos', light_pos)

    glActiveTexture(GL_TEXTURE1)
    program.setInt('mat.diffuseMap', 1)
    texture.bind()

    glActiveTexture(GL_TEXTURE2)
    program.setInt('mat.normalMap', 2)
    normal.bind()

    glActiveTexture(GL_TEXTURE3)
    program.setInt('mat.specularMap', 3)
    specular.bind()

    glActiveTexture(GL_TEXTURE4)
    program.setInt('mat.depthMap', 4)
    depth.bind()
    program.setFloat('mat.shininess', 128)
    program.setFloat('mat.heightScale', 0.12)

    obj.draw_multiple(program)


def main():
    global hdrbuffer, blurbuffer, cube, square, width, height

    if not glfw.init():
        print('Failed to initialize GLFW.')
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.SAMPLES, config['sampling_level'])

    if config['fullscreen']:
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        width, height = mode.size.width, mode.size.height
        window = glfw.create_window(mode.size.width,
                                    mode.size.height,
                                    config['app_name'],
                                    glfw.get_primary_monitor(),
                                    None)
    else:
        window = glfw.create_window(width, height, config['app_name'], None, None)
    if not window:
        print('Failed to create GLFW Window.')
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_framebuffer_size_callback(window, resize_callback)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # basic_program = get_linked_program('resources/shaders/vert.vs', 'resources/shaders/frag.fs')
    program = get_linked_program('resources/shaders/vert.vs', 'resources/shaders/frag.fs')
    depth_program = get_linked_program('resources/shaders/shadow_depth.vs', 'resources/shaders/shadow_depth.fs')
    blur_program = get_linked_program('resources/shaders/blur.vs', 'resources/shaders/blur.fs')
    hdr_program = get_linked_program('resources/shaders/hdr.vs', 'resources/shaders/hdr.fs')

    blur_program.use()
    blur_program.setInt('image', 0)

    hdr_program.use()
    hdr_program.setInt('sceneMap', 0)
    hdr_program.setInt('bloomMap', 1)

    hdrbuffer = HDRBuffer()
    hdrbuffer.create(width, height)
    blurbuffer = BlurBuffer()
    blurbuffer.create(width, height)

    bloom = Bloom(hdrbuffer, hdr_program, blurbuffer, blur_program)

    light_pos = glm.vec3(0.5, 0.5, 0.5)
    perspective = glm.perspective(45, width / height, config['near_plane'], config['far_plane'])

    cam_angles = [[0, 315, -45], [0, 225, -45], [0, 135, -45], [0, 45, -45]]
    cam_shapes = [Model('resources/models/camera.json', cam_angles[c]) for c in range(4)]
    square = Model('resources/models/square.json')
    cube = Model('resources/models/cube.json')
    texture = load_texture_2d('resources/textures/diffuse.jpg')
    texture_grid = load_texture_2d('resources/textures/diffuse_grid.jpg')
    normal = load_texture_2d('resources/textures/normal.jpg')
    normal_grid = load_texture_2d('resources/textures/normal_grid.jpg')
    specular = load_texture_2d('resources/textures/specular.jpg')
    specular_grid = load_texture_2d('resources/textures/specular_grid.jpg')
    depth = load_texture_2d('resources/textures/depth.jpg')
    depth_grid = load_texture_2d('resources/textures/depth_grid.jpg')

    grid_positions = generate_grid(config['world_width'], config['world_width'])
    square.set_multiple_positions(grid_positions)

    cam_positions = get_cam_pos()
    for c, cam_pos in enumerate(cam_positions):
        cam_shapes[c].set_multiple_positions([cam_pos])

    last_time = glfw.get_time()
    last_voxel_update = glfw.get_time()
    first_iteration = True
    while not glfw.window_should_close(window):
        if config['debug_mode']:
            print(glGetError())

        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time
        if first_iteration or current_time - last_voxel_update > 2:
            first_iteration = False
            # block_positions = generate_voxel_positions(config['world_width'], config['world_height'], config['world_width'])
            # cube.set_multiple_positions(block_positions)
            # block_positions.clear()
            last_voxel_update = current_time

        move_input(window, delta_time)
        # print(camera.position)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.2, 0.8, 1)
        # glClearColor(0, 0, 0, 1)

        square.draw_multiple(depth_program)
        cube.draw_multiple(depth_program)
        for cam in cam_shapes:
            cam.draw_multiple(depth_program)

        hdrbuffer.bind()
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #
        draw_objs(cube, program, perspective, light_pos, texture, normal, specular, depth)
        draw_objs(square, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)
        for cam in cam_shapes:
            draw_objs(cam, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)

        hdrbuffer.unbind()
        hdrbuffer.finalize()

        bloom.draw_processed_scene()

        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()


def resize_callback(window, w, h):
    if h > 0:
        global width, height, hdrbuffer, blurbuffer
        width, height = w, h
        glm.perspective(45, width / height, config['near_plane'], config['far_plane'])
        hdrbuffer.delete()
        hdrbuffer.create(width, height)
        blurbuffer.delete()
        blurbuffer.create(width, height)


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, glfw.TRUE)
    if key == glfw.KEY_G and action == glfw.PRESS:
        global cube
        positions = generate_voxel_positions(config['world_width'], config['world_height'], config['world_width'])
        cube.set_multiple_positions(positions)


def mouse_move(win, pos_x, pos_y):
    global firstTime, camera, lastPosX, lastPosY
    if firstTime:
        lastPosX = pos_x
        lastPosY = pos_y
        firstTime = False

    camera.rotate(pos_x - lastPosX, lastPosY - pos_y)
    lastPosX = pos_x
    lastPosY = pos_y


def move_input(win, time):
    if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
        camera.move_top(time)
    if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
        camera.move_bottom(time)
    if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
        camera.move_left(time)
    if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
        camera.move_right(time)


if __name__ == '__main__':
    main()
