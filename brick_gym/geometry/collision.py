import numpy

from renderpy.frame_buffer import FrameBufferWrapper
from renderpy.camera import orthographic_matrix
from renderpy.image import save_image, save_depth

def check_collision(
        scene,
        instance,
        snap_transform,
        instance_snap_gender,
        resolution=(512,512),
        frame_buffer=None,
        max_intersection=4,
    ):
    
    #===========================================================================
    # setup
    #---------------------------------------------------------------------------
    # make sure the scene is renderable
    assert scene.renderable
    #---------------------------------------------------------------------------
    # build a renderpy frame buffer if a shared one was not specified
    if frame_buffer is None:
        frame_buffer = FrameBufferWrapper(
                resolution[0],
                resolution[1],
                anti_alias=False)
    #---------------------------------------------------------------------------
    # store the camera info and which bricks are hidden
    original_camera_pose = scene.get_camera_pose()
    original_camera_projection = scene.get_projection()
    hidden_instances = {i : scene.instance_hidden(i) for i in scene.instances}
    #---------------------------------------------------------------------------
    # compute the camera distance, clipping plane and the orthgraphic width
    camera_distance = 500 # TMP
    orthographic_width = 100 # TMP
    orthographic_height = 100 # TMP
    near_clip = 1 # TMP
    far_clip = 2000 # TMP
    if instance_snap_gender.upper() == 'M':
        brick_direction = 1
    else:
        brick_direction = -1
    
    #scene.set_ambient_color((1,1,1))
    scene.load_image_light('default', texture_directory='grey_cube')
    scene.set_active_image_light('default')
    
    #===========================================================================
    # render the scene depth map
    #---------------------------------------------------------------------------
    # show everything except for the target instance
    scene.show_all_instances()
    scene.hide_instance(instance)
    #---------------------------------------------------------------------------
    # setup the camera
    camera_transform = snap_transform.copy()
    render_axis = snap_transform[:3,1]
    render_axis /= numpy.linalg.norm(render_axis)
    camera_transform[:3,3] += render_axis * camera_distance * brick_direction
    y_to_neg_z = numpy.array([
            [1, 0, 0, 0],
            [0, 0,-1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])
    camera_transform = numpy.dot(camera_transform, y_to_neg_z)
    scene.set_camera_pose(numpy.linalg.inv(camera_transform))
    orthographic_projection = orthographic_matrix(
            l = -orthographic_width,
            r = orthographic_width,
            b = -orthographic_height,
            t = orthographic_height,
            n = near_clip,
            f = far_clip)
    scene.set_projection(orthographic_projection)
    #---------------------------------------------------------------------------
    # render
    frame_buffer.enable()
    scene.mask_render()
    scene_mask = frame_buffer.read_pixels()
    scene_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    scene.color_render()
    scene_color = frame_buffer.read_pixels()
    
    #===========================================================================
    # render the instance depth map
    #---------------------------------------------------------------------------
    # hide everything except for the target instance
    scene.hide_all_instances()
    scene.show_instance(instance)
    #---------------------------------------------------------------------------
    # setup the camera
    camera_transform = snap_transform.copy()
    camera_transform[:3,3] -= render_axis * camera_distance * brick_direction
    y_to_pos_z = numpy.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])
    camera_transform = numpy.dot(camera_transform, y_to_pos_z)
    scene.set_camera_pose(numpy.linalg.inv(camera_transform))
    scene.set_projection(orthographic_projection)
    #---------------------------------------------------------------------------
    # render
    frame_buffer.enable()
    scene.mask_render()
    instance_mask = frame_buffer.read_pixels()
    instance_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=orthographic_projection)
    
    scene.color_render()
    instance_color = frame_buffer.read_pixels()
    
    #===========================================================================
    # restore the previous camera and hidden state
    scene.set_camera_pose(original_camera_pose)
    scene.set_projection(original_camera_projection)
    for instance, hidden in hidden_instances.items():
        if hidden:
            scene.hide_instance(instance)
        else:
            scene.show_instance(instance)
    
    #===========================================================================
    # check collision
    save_image(scene_mask, './tmp_scene_mask.png')
    save_image(instance_mask, './tmp_instance_mask.png')
    save_depth(scene_depth_map, './tmp_scene_depth.npy')
    save_depth(instance_depth_map, './tmp_instance_depth.npy')
    save_image(scene_color, './tmp_scene_color.png')
    save_image(instance_color, './tmp_instance_color.png')
    
    valid_pixels = numpy.sum(instance_mask != 0, axis=-1) != 0
    
    scene_depth_map -= camera_distance
    scene_depth_map *= -1.
    instance_depth_map -= camera_distance
    offset = (scene_depth_map - instance_depth_map).reshape(resolution)
    offset *= valid_pixels
    
    collision = numpy.max(offset) > max_intersection
    
    return collision
